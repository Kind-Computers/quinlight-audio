// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(
    clippy::collapsible_if,
    clippy::match_overlapping_arm,
    clippy::needless_borrow,
    clippy::too_many_arguments
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

#[cfg(test)]
use std::sync::atomic::AtomicI32;

use crate::openmpt::{DEFAULT_INTERPOLATION_FILTER_LENGTH, Module};
use crate::remaster::{
    self, CleanupSettings, RemasterEngine, RemasterOutput, RemasterStatus, UpscaleMode,
    extract_sample_jobs,
};

struct RemasterOutcome {
    replaced: usize,
    total: usize,
    /// Samples where AI consensus was reached (2+ engines above score floor) and applied.
    consensus_replaced: usize,
    /// Sample indices where no AI engine produced any output at all.
    no_output_indices: Vec<i32>,
    /// Sample indices where AI ran but consensus was not reached (< 2 usable engines).
    no_consensus_indices: Vec<i32>,
    /// Number of AI engines available when this mod was processed.
    engines_available: usize,
    /// Samples already at >= 48 kHz (skipped by the AI pipeline but still
    /// benefit from the new mixing engine).
    existing_hifi_samples: usize,
}

/// Exit-code-aware error type for batch conversion.
///
/// `Fatal` → exit 1 (engine/subprocess broken, stop the batch script).
/// `QualityGate` → exit 2 (mod not remastered, but engine is OK — script may continue).
#[derive(Debug)]
pub enum ConvertError {
    Fatal(String),
    QualityGate(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fatal(msg) | Self::QualityGate(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for ConvertError {}

fn format_indices(indices: &[i32]) -> String {
    indices
        .iter()
        .map(|i| format!("#{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Format a sample rate in Hz as a human-readable kHz string for filenames.
/// Handles fractional rates like 44100 → "44.1Khz" and even rates like 96000 → "96Khz".
pub(crate) fn format_rate_khz(rate_hz: u32) -> String {
    if rate_hz.is_multiple_of(1000) {
        format!("{}Khz", rate_hz / 1000)
    } else {
        format!("{:.1}Khz", rate_hz as f64 / 1000.0)
    }
}

#[cfg(test)]
static LAST_RENDER_INTERPOLATION_FILTER: AtomicI32 = AtomicI32::new(i32::MIN);

/// AAC encoders cap at 96 kHz.
const AAC_MAX_RATE: u32 = 96_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConvertOutputKind {
    Original,
    QuinlightRemastered48k,
}

impl ConvertOutputKind {
    fn nominal_rate(&self) -> u32 {
        match self {
            Self::QuinlightRemastered48k | Self::Original => 96_000,
        }
    }

    fn effective_rate(&self, format: &str) -> u32 {
        let rate = self.nominal_rate();
        match format {
            "m4a" => rate.min(AAC_MAX_RATE),
            _ => rate,
        }
    }
}

pub(crate) const fn default_render_interpolation_filter() -> i32 {
    DEFAULT_INTERPOLATION_FILTER_LENGTH
}

#[cfg(test)]
fn record_render_interpolation_filter(interpolation_filter: i32) {
    LAST_RENDER_INTERPOLATION_FILTER.store(interpolation_filter, Ordering::Relaxed);
}

#[cfg(test)]
fn take_recorded_render_interpolation_filter() -> Option<i32> {
    let recorded = LAST_RENDER_INTERPOLATION_FILTER.swap(i32::MIN, Ordering::Relaxed);
    (recorded != i32::MIN).then_some(recorded)
}

/// Resolve input bytes: if the path is an archive, extract the specified (or first) module.
fn resolve_input(input: &Path, archive_file: Option<&str>) -> Result<Vec<u8>, String> {
    if crate::archive::is_archive(input) {
        let entries = crate::archive::list_modules_in_archive(input)?;
        if entries.is_empty() {
            return Err("No module files found in archive".into());
        }
        let entry_path = if let Some(name) = archive_file {
            // Find the requested file (match by filename or full path)
            entries
                .iter()
                .find(|e| e.path == name || e.filename == name)
                .map(|e| e.path.as_str())
                .ok_or_else(|| {
                    let available: Vec<&str> =
                        entries.iter().map(|e| e.filename.as_str()).collect();
                    format!(
                        "'{name}' not found in archive. Available: {}",
                        available.join(", ")
                    )
                })?
        } else {
            eprintln!("Using first module: {}", entries[0].filename);
            &entries[0].path
        };
        crate::archive::extract_from_archive(input, entry_path)
    } else {
        crate::archive::read_module_file(input)
    }
}

fn render_module_to_output(
    module: &mut Module,
    output: &Path,
    format: &str,
    stereo_separation: i32,
    agc_enabled: bool,
    sample_rate: u32,
    hrtf_mix: i32,
) -> Result<(), String> {
    render_module_to_output_with_interpolation(
        module,
        output,
        format,
        stereo_separation,
        default_render_interpolation_filter(),
        agc_enabled,
        sample_rate,
        hrtf_mix,
    )
}

fn render_module_to_output_with_interpolation(
    module: &mut Module,
    output: &Path,
    format: &str,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    hrtf_mix: i32,
) -> Result<(), String> {
    #[cfg(test)]
    record_render_interpolation_filter(interpolation_filter);

    let info = module.info();
    let metadata = crate::render::AudioMetadata {
        title: info.title.clone(),
        artist: info.artist.clone(),
        album: "Quinlight Audio".into(),
    };
    match format {
        "flac" => crate::render::render_live_module_to_flac(
            module,
            output,
            stereo_separation,
            interpolation_filter,
            agc_enabled,
            sample_rate,
            hrtf_mix,
            &metadata,
        ),
        "m4a" => crate::render::render_live_module_to_aac(
            module,
            output,
            stereo_separation,
            interpolation_filter,
            agc_enabled,
            sample_rate,
            hrtf_mix,
            &metadata,
        ),
        _ => Err(format!("Unknown format: {format}. Use 'flac' or 'm4a'.")),
    }
}

/// Build the output filename from the stem, format extension, output kind, and the
/// **effective** render rate (which may be lower than the kind's nominal rate — e.g.
/// AAC caps at 96 kHz).
fn converted_output_name(
    stem: &str,
    format: &str,
    output_kind: ConvertOutputKind,
    render_rate_hz: u32,
) -> String {
    let stem = match output_kind {
        ConvertOutputKind::Original => stem.to_string(),
        ConvertOutputKind::QuinlightRemastered48k => {
            format!(
                "{stem}-Quinlight-Audio-Remastered-{}",
                format_rate_khz(render_rate_hz)
            )
        }
    };
    format!("{stem}.{format}")
}

fn resolve_requested_engines(
    requested: &[String],
    available: &[&str],
) -> Result<Vec<String>, String> {
    let mut enabled = Vec::new();
    let mut missing = Vec::new();

    for requested_name in requested {
        if enabled
            .iter()
            .any(|name: &String| name.eq_ignore_ascii_case(requested_name))
            || missing
                .iter()
                .any(|name: &String| name.eq_ignore_ascii_case(requested_name))
        {
            continue;
        }

        if let Some(&available_name) = available
            .iter()
            .find(|available_name| available_name.eq_ignore_ascii_case(requested_name))
        {
            enabled.push(available_name.to_string());
        } else {
            missing.push(requested_name.clone());
        }
    }

    if missing.is_empty() {
        Ok(enabled)
    } else {
        let available_display = if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        };
        Err(format!(
            "Requested Quinlight engine(s) not available: {}. Available engines: {available_display}",
            missing.join(", ")
        ))
    }
}

pub(crate) fn detect_remaster_engine(
    requested_engines: &[String],
) -> Result<RemasterEngine, String> {
    if requested_engines.is_empty() {
        let engine = RemasterEngine::detect();
        if engine.is_available() {
            return Ok(engine);
        }

        return Err(format!(
            "No AI remastering engines found.\n\n{}",
            remaster::install_instructions()
        ));
    }

    let detected = RemasterEngine::detect();
    let enabled = resolve_requested_engines(requested_engines, &detected.available_engine_names())?;
    let engine = RemasterEngine::detect_with_fallback(&enabled);
    if engine.is_available() {
        Ok(engine)
    } else {
        Err(format!(
            "Requested Quinlight engine(s) not available: {}",
            requested_engines.join(", ")
        ))
    }
}

fn snapshot_effect_params_by_sample(
    module: &Module,
    originals: &[remaster::OriginalSample],
) -> std::collections::HashMap<i32, Vec<remaster::SavedEffectParam>> {
    originals
        .iter()
        .map(|original| {
            (
                original.index,
                remaster::save_effect_params(module, original.index),
            )
        })
        .collect()
}

/// Apply Quinlight sample replacements directly to an already-loaded module.
fn remaster_module_in_place(
    module: &mut Module,
    engine: &RemasterEngine,
    mode: UpscaleMode,
    full_parallel: bool,
    cleanup_settings: CleanupSettings,
    log_prefix: &str,
    ddim_steps: u32,
    cancel_flag: &std::sync::atomic::AtomicBool,
) -> Result<RemasterOutcome, String> {
    if !engine.is_available() {
        return Err(format!(
            "No AI remastering engines found.\n\n{}",
            remaster::install_instructions()
        ));
    }

    let work_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let min_dur = engine.min_duration_secs();
    let engines_available = engine.engine_count();
    let existing_hifi_samples = remaster::count_high_fidelity_samples(module);
    let originals = remaster::read_raw_samples(module);
    let jobs = extract_sample_jobs(
        &originals,
        work_dir.path(),
        min_dur,
        cleanup_settings,
        &cancel_flag,
    )?;

    if jobs.is_empty() {
        eprintln!("{log_prefix}No samples to remaster (all already at 48kHz or higher)");
        return Ok(RemasterOutcome {
            replaced: 0,
            total: 0,
            consensus_replaced: 0,
            no_output_indices: vec![],
            no_consensus_indices: vec![],
            engines_available,
            existing_hifi_samples,
        });
    }

    let total = jobs.len();
    let job_indices: Vec<i32> = jobs.iter().map(|j| j.index).collect();
    eprintln!("{log_prefix}Remastering {total} samples...");

    let original_rates: std::collections::HashMap<i32, i32> =
        originals.iter().map(|o| (o.index, o.rate)).collect();
    let saved_effects_by_sample = snapshot_effect_params_by_sample(module, &originals);

    let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
    let (result_tx, result_rx) = crossbeam_channel::unbounded();

    let engine_ref = engine;
    let mut replaced = 0;
    let mut consensus_replaced = 0;
    let mut no_output_indices = Vec::new();
    let mut no_consensus_indices = Vec::new();
    std::thread::scope(|s| {
        s.spawn(move || {
            let _ = engine_ref.remaster_samples(
                jobs,
                work_dir,
                &progress_tx,
                &result_tx,
                mode,
                &cancel_flag,
                ddim_steps,
                false,
                full_parallel,
            );
        });

        let progress_handle = s.spawn(move || {
            for status in progress_rx {
                if let RemasterStatus::Processing {
                    current,
                    total,
                    sample_name,
                } = status
                {
                    eprintln!("{log_prefix}[{current}/{total}] {sample_name}");
                }
            }
        });

        let mut best_results = std::collections::HashMap::<i32, remaster::SampleResult>::new();
        for output in result_rx {
            record_best_final_result(&mut best_results, output);
        }

        // Classify results before consuming best_results.
        no_consensus_indices = best_results
            .values()
            .filter(|r| remaster::is_no_consensus_result(&r.engine_name))
            .map(|r| r.index)
            .collect();
        let result_indices: std::collections::HashSet<i32> = best_results.keys().copied().collect();
        no_output_indices = job_indices
            .iter()
            .filter(|idx| !result_indices.contains(idx))
            .copied()
            .collect();

        for result in best_results.into_values() {
            if !should_apply_final_result(&result) {
                continue;
            }
            let orig_rate = original_rates.get(&result.index).copied().unwrap_or(0);
            let saved_effects = saved_effects_by_sample
                .get(&result.index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if let Err(error) = remaster::apply_sample_replacement(
                module,
                result.index,
                &result.data,
                result.length_frames,
                result.channels,
                result.sample_rate_hz,
                orig_rate,
                saved_effects,
            ) {
                eprintln!(
                    "{log_prefix}Failed to apply remastered sample {}: {error}",
                    result.index + 1
                );
            } else {
                replaced += 1;
                consensus_replaced += 1;
            }
        }

        let _ = progress_handle.join();
    });

    if replaced == 0 {
        eprintln!("{log_prefix}No samples were successfully remastered");
    } else {
        eprintln!("{log_prefix}{replaced}/{total} samples remastered");
    }

    Ok(RemasterOutcome {
        replaced,
        total,
        consensus_replaced,
        no_output_indices,
        no_consensus_indices,
        engines_available,
        existing_hifi_samples,
    })
}

pub fn run_render(
    input: &Path,
    output: &Path,
    format: &str,
    stereo_separation: i32,
    agc_enabled: bool,
    archive_file: Option<&str>,
    sample_rate: u32,
    hrtf_mix: i32,
) -> Result<(), String> {
    let file_data = resolve_input(input, archive_file)?;
    let mut module = Module::from_memory(&file_data)?;

    eprintln!("Rendering {}...", input.display());

    render_module_to_output(
        &mut module,
        output,
        format,
        stereo_separation,
        agc_enabled,
        sample_rate,
        hrtf_mix,
    )?;

    eprintln!("Saved: {}", output.display());
    Ok(())
}

/// Collect all module files in a directory, optionally recursing into subdirectories.
fn collect_module_files(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];

    while let Some(current) = dirs.pop() {
        let entries = std::fs::read_dir(&current)
            .map_err(|e| format!("Failed to read directory {}: {e}", current.display()))?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    dirs.push(path);
                }
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if crate::archive::is_module_file(name) {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

#[allow(clippy::too_many_arguments)]
pub fn run_convert(
    input: &Path,
    output_dir: Option<&Path>,
    formats: &[String],
    stereo_separation: Option<i32>,
    agc_enabled: bool,
    recursive: bool,
    remaster: bool,
    requested_engines: &[String],
    mode: UpscaleMode,
    full_parallel: bool,
    cleanup_settings: CleanupSettings,
    ddim_steps: u32,
    sample_rate_override: Option<u32>,
    shutdown_flag: &std::sync::atomic::AtomicBool,
    hrtf_mix: i32,
) -> Result<(), ConvertError> {
    // Accept a single file or a directory of files
    let (mut files, base_dir) = if input.is_file() {
        let base = input.parent().unwrap_or(input).to_path_buf();
        (vec![input.to_path_buf()], base)
    } else if input.is_dir() {
        let f = collect_module_files(input, recursive).map_err(ConvertError::Fatal)?;
        (f, input.to_path_buf())
    } else {
        return Err(ConvertError::Fatal(format!(
            "{} is not a file or directory",
            input.display()
        )));
    };

    // Validate formats
    for f in formats {
        if f != "flac" && f != "m4a" {
            return Err(ConvertError::Fatal(format!(
                "Unknown format: {f}. Use 'flac' or 'm4a'."
            )));
        }
    }

    let out_dir = output_dir.unwrap_or(&base_dir);
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir)
            .map_err(|e| ConvertError::Fatal(format!("Failed to create output directory: {e}")))?;
    }

    fastrand::shuffle(&mut files);
    if files.is_empty() {
        eprintln!("No module files found in {}", input.display());
        return Ok(());
    }

    let total = files.len();
    eprintln!("Found {total} module files");

    let requested_output_kind = if remaster {
        ConvertOutputKind::QuinlightRemastered48k
    } else {
        ConvertOutputKind::Original
    };

    let remaster_engine = if !remaster {
        None
    } else {
        Some(detect_remaster_engine(requested_engines).map_err(ConvertError::Fatal)?)
    };

    let mut succeeded = 0;
    let mut skipped = 0;
    let mut fatal_count = 0;
    let mut gate_count = 0;

    for (i, path) in files.iter().enumerate() {
        if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!("Batch conversion cancelled");
            break;
        }
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let rel = path.strip_prefix(&base_dir).unwrap_or(path);
        // Preserve subdirectory structure: input/jazz/cool.s3m → output/jazz/cool.flac
        let rel_parent = rel.parent().unwrap_or(std::path::Path::new(""));
        let sub_dir = out_dir.join(rel_parent);
        let rel = rel.display().to_string();

        // Check if all outputs already exist (using effective rate for filename)
        let all_exist = formats.iter().all(|fmt| {
            let rate =
                sample_rate_override.unwrap_or_else(|| requested_output_kind.effective_rate(fmt));
            let out_path = sub_dir.join(converted_output_name(
                &stem,
                fmt,
                requested_output_kind,
                rate,
            ));
            out_path.exists()
        });
        if all_exist {
            eprintln!("[{}/{}] Skipping {rel} (already converted)", i + 1, total);
            skipped += 1;
            continue;
        }

        eprintln!("[{}/{}] Converting {rel}", i + 1, total);

        let file_data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  Error reading file: {e}");
                skipped += 1;
                continue;
            }
        };

        let mut module = match Module::from_memory(&file_data) {
            Ok(module) => module,
            Err(e) => {
                eprintln!("  Error loading module: {e}");
                skipped += 1;
                continue;
            }
        };

        let mut output_kind = ConvertOutputKind::Original;
        if let Some(engine) = remaster_engine.as_ref() {
            match remaster_module_in_place(
                &mut module,
                engine,
                mode,
                full_parallel,
                cleanup_settings,
                "  ",
                ddim_steps,
                shutdown_flag,
            ) {
                Ok(outcome) => {
                    if !outcome.no_output_indices.is_empty() {
                        eprintln!(
                            "  FATAL: engine(s) produced no output for sample(s) {}",
                            format_indices(&outcome.no_output_indices),
                        );
                        fatal_count += 1;
                        break;
                    }
                    let mixing_engine_qualifies =
                        outcome.engines_available >= 2 && outcome.existing_hifi_samples >= 1;
                    if !outcome.no_consensus_indices.is_empty() {
                        if outcome.consensus_replaced > 0 {
                            eprintln!(
                                "  Note: sample(s) {} kept the original sample \
                                 (consensus not reached), {}/{} samples AI-remastered",
                                format_indices(&outcome.no_consensus_indices),
                                outcome.consensus_replaced,
                                outcome.total,
                            );
                        } else if mixing_engine_qualifies {
                            eprintln!(
                                "  Note: AI consensus not reached, but mod has {} sample(s) \
                                 already at >=48 kHz and {} engines were available — \
                                 accepting as Quinlight-remastered via the mixing engine",
                                outcome.existing_hifi_samples, outcome.engines_available,
                            );
                        } else {
                            eprintln!(
                                "  AI quality gate: no samples achieved consensus \
                                 (requires 2+ engines above score floor)",
                            );
                            gate_count += 1;
                            continue;
                        }
                    }
                    if outcome.replaced > 0 || mixing_engine_qualifies {
                        output_kind = ConvertOutputKind::QuinlightRemastered48k;
                    }
                }
                Err(e) => {
                    if remaster::is_cancelled_error(&e) || shutdown_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    eprintln!("  FATAL: {e}");
                    fatal_count += 1;
                    break;
                }
            }
        }

        // Render to each requested format
        if !rel_parent.as_os_str().is_empty() {
            std::fs::create_dir_all(&sub_dir).map_err(|e| {
                ConvertError::Fatal(format!(
                    "Failed to create output subdirectory {}: {e}",
                    sub_dir.display()
                ))
            })?;
        }
        for fmt in formats {
            let render_rate =
                sample_rate_override.unwrap_or_else(|| output_kind.effective_rate(fmt));

            let out_path =
                sub_dir.join(converted_output_name(&stem, fmt, output_kind, render_rate));
            if out_path.exists() {
                eprintln!("  Skipping {} (already exists)", out_path.display());
                continue;
            }
            let effective_sep = stereo_separation.unwrap_or_else(|| {
                crate::openmpt::effective_stereo_separation(
                    path,
                    crate::openmpt::DEFAULT_STEREO_SEPARATION_PERCENT,
                )
            });
            let result = render_module_to_output(
                &mut module,
                &out_path,
                fmt,
                effective_sep,
                agc_enabled,
                render_rate,
                hrtf_mix,
            );

            match result {
                Ok(()) => eprintln!("  Saved {}", out_path.display()),
                Err(e) => {
                    eprintln!("  FATAL: Error rendering {fmt}: {e}");
                    fatal_count += 1;
                    break;
                }
            }
        }

        succeeded += 1;
    }

    eprintln!(
        "\nDone: {succeeded} converted, {skipped} skipped, \
         {gate_count} failed quality gate, {fatal_count} fatal"
    );
    if fatal_count > 0 {
        Err(ConvertError::Fatal(format!(
            "{fatal_count} file(s) hit fatal engine failure"
        )))
    } else if gate_count > 0 {
        Err(ConvertError::QualityGate(format!(
            "{gate_count} file(s) failed AI quality gate"
        )))
    } else {
        Ok(())
    }
}

fn final_result(output: RemasterOutput) -> Option<remaster::SampleResult> {
    match output {
        RemasterOutput::Final(result) if !result.data.is_empty() => Some(result),
        RemasterOutput::Candidate(_) | RemasterOutput::Final(_) => None,
    }
}

fn record_best_final_result(
    finals: &mut std::collections::HashMap<i32, remaster::SampleResult>,
    output: RemasterOutput,
) {
    let Some(result) = final_result(output) else {
        return;
    };

    match finals.get_mut(&result.index) {
        Some(existing) if existing.sample_rate_hz > result.sample_rate_hz => {}
        Some(existing) => *existing = result,
        None => {
            finals.insert(result.index, result);
        }
    }
}

fn should_apply_final_result(result: &remaster::SampleResult) -> bool {
    !remaster::is_no_consensus_result(&result.engine_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_FIXTURE: &str = "mods/2ND_PM.S3M";
    const XM_FIXTURE: &str = "openmpt/test/test.xm";
    const SAMPLE_INDEX: i32 = 32;
    const COMMAND_NOTE: i32 = 0;
    const COMMAND_INSTRUMENT: i32 = 1;
    const COMMAND_EFFECT: i32 = 3;
    const COMMAND_PARAMETER: i32 = 5;
    const CMD_OFFSET: u8 = 10;

    fn sample_result(name: &str, data: Vec<f64>) -> remaster::SampleResult {
        remaster::SampleResult {
            index: 3,
            data,
            length_frames: 4,
            channels: 1,
            sample_rate_hz: 48_000,
            engine_name: name.to_string(),
            discovered_loops: None,
        }
    }

    fn original_signal() -> Vec<f64> {
        (0..4096)
            .map(|i| (i as f64 * 220.0 * 2.0 * std::f64::consts::PI / 48_000.0).sin())
            .collect()
    }

    fn xm_reference_fixture() -> (Module, u8) {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mapped_note = (1u8..=120)
            .find(|&note| module.instrument_sample_for_note(0, note) == Some(1))
            .expect("Fixture instrument should map at least one note to sample 2");

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, mapped_note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_OFFSET));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, 0x10));

        let original = original_signal();
        let original_rate = 16_000;
        assert!(module.replace_sample_data(1, &original, original.len() as i64, 1, original_rate,));

        let expected_param = ((0x10u32 * 48_000) / original_rate as u32).min(255) as u8;
        (module, expected_param)
    }

    #[test]
    fn final_result_ignores_candidate_outputs() {
        let output = RemasterOutput::Candidate(sample_result("AudioSR", vec![0.1, 0.2]));
        assert!(final_result(output).is_none());
    }

    #[test]
    fn final_result_returns_non_empty_final_outputs() {
        let expected = sample_result("Quinlight Audio (A+L)", vec![0.1, 0.2, 0.3, 0.4]);
        let actual = final_result(RemasterOutput::Final(expected.clone()));
        assert_eq!(
            actual.as_ref().map(|result| result.engine_name.as_str()),
            Some("Quinlight Audio (A+L)")
        );
        assert_eq!(actual.map(|result| result.data), Some(expected.data));
    }

    #[test]
    fn bare_quinlight_final_keeps_original_sample_in_batch_mode() {
        assert!(
            !should_apply_final_result(&sample_result("Quinlight Audio", vec![0.1, 0.2])),
            "no-consensus finals should leave the original sample untouched in batch mode",
        );
        assert!(
            should_apply_final_result(&sample_result("Quinlight Audio (A+L)", vec![0.1, 0.2])),
            "consensus finals should still be applied in batch mode",
        );
    }

    #[test]
    fn record_best_final_result_keeps_only_highest_rate_per_sample() {
        let mut finals = std::collections::HashMap::new();
        let mut low = sample_result("Quinlight Audio (A)", vec![0.1, 0.2]);
        low.index = 9;
        low.sample_rate_hz = 44_100;
        let mut high = low.clone();
        high.sample_rate_hz = 48_000;

        record_best_final_result(&mut finals, RemasterOutput::Final(low));
        record_best_final_result(
            &mut finals,
            RemasterOutput::Candidate(sample_result("AudioSR", vec![0.0; 2])),
        );
        record_best_final_result(&mut finals, RemasterOutput::Final(high.clone()));

        assert_eq!(finals.len(), 1);
        assert_eq!(
            finals.get(&9).map(|result| result.sample_rate_hz),
            Some(48_000)
        );
        assert_eq!(
            finals.get(&9).map(|result| result.data.clone()),
            Some(high.data)
        );
    }

    #[test]
    fn record_best_final_result_prefers_later_final_at_equal_rate() {
        // When the fallback wave refines a primary consensus, two 48 kHz
        // Finals land for the same sample_index — the later one has more
        // contributors and must win.
        let mut finals = std::collections::HashMap::new();
        let mut primary_final = sample_result("Quinlight Audio (A+L)", vec![0.1, 0.2]);
        primary_final.index = 7;
        primary_final.sample_rate_hz = 48_000;
        let mut fallback_final = sample_result("Quinlight Audio (A+L+F)", vec![0.3, 0.4]);
        fallback_final.index = 7;
        fallback_final.sample_rate_hz = 48_000;

        record_best_final_result(&mut finals, RemasterOutput::Final(primary_final));
        record_best_final_result(&mut finals, RemasterOutput::Final(fallback_final.clone()));

        let stored = finals.get(&7).expect("final recorded");
        assert_eq!(stored.engine_name, "Quinlight Audio (A+L+F)");
        assert_eq!(stored.data, fallback_final.data);
    }

    #[test]
    fn converted_output_name_uses_render_rate_suffix() {
        let k = ConvertOutputKind::QuinlightRemastered48k;
        assert_eq!(
            converted_output_name("2ND_PM", "m4a", k, 96_000),
            "2ND_PM-Quinlight-Audio-Remastered-96Khz.m4a"
        );
        assert_eq!(
            converted_output_name("2ND_PM", "flac", k, 48_000),
            "2ND_PM-Quinlight-Audio-Remastered-48Khz.flac"
        );
        assert_eq!(
            converted_output_name("2ND_PM", "flac", k, 192_000),
            "2ND_PM-Quinlight-Audio-Remastered-192Khz.flac"
        );
        assert_eq!(
            converted_output_name("2ND_PM", "flac", k, 44_100),
            "2ND_PM-Quinlight-Audio-Remastered-44.1Khz.flac"
        );
        assert_eq!(
            converted_output_name("2ND_PM", "flac", k, 88_200),
            "2ND_PM-Quinlight-Audio-Remastered-88.2Khz.flac"
        );
        let k = ConvertOutputKind::Original;
        assert_eq!(
            converted_output_name("2ND_PM", "m4a", k, 96_000),
            "2ND_PM.m4a"
        );
    }

    #[test]
    fn resolve_requested_engines_keeps_requested_subset_with_canonical_names() {
        let requested = vec![
            "lavasr".to_string(),
            "AudioSR".to_string(),
            "LavaSR".to_string(),
        ];
        let available = ["AudioSR", "LavaSR", "FLowHigh"];

        let resolved =
            resolve_requested_engines(&requested, &available).expect("Subset should resolve");

        assert_eq!(resolved, vec!["LavaSR".to_string(), "AudioSR".to_string()]);
    }

    #[test]
    fn resolve_requested_engines_rejects_missing_engines() {
        let requested = vec!["AudioSR".to_string(), "LavaSR".to_string()];
        let available = ["AudioSR"];

        let error = resolve_requested_engines(&requested, &available)
            .expect_err("Missing engines should be rejected");

        assert!(error.contains("LavaSR"));
        assert!(error.contains("AudioSR"));
    }

    #[test]
    fn batch_snapshot_repatches_from_original_xm_effects() {
        let (mut module, expected_param) = xm_reference_fixture();
        let originals = remaster::read_raw_samples(&mut module);
        let saved_effects_by_sample = snapshot_effect_params_by_sample(&module, &originals);
        let original = originals
            .into_iter()
            .find(|original| original.index == 1)
            .expect("fixture should yield a remasterable target sample");
        let replacement: Vec<f64> = (0..12_288)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                0.5 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
            })
            .collect();
        let saved_effects = saved_effects_by_sample
            .get(&original.index)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        assert_eq!(saved_effects, &[(0, 0, 0, 0x10)]);
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, expected_param));

        remaster::apply_sample_replacement(
            &mut module,
            original.index,
            &replacement,
            replacement.len() as i64 / original.channels as i64,
            original.channels,
            48_000,
            original.rate,
            saved_effects,
        )
        .expect("Batch apply should succeed");

        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Batch snapshot should restore original XM offsets before repatching",
        );
    }

    #[test]
    fn render_module_to_output_uses_live_module_state() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut original_module = Module::from_memory(&data).expect("Failed to load module");
        let mut edited_module = Module::from_memory(&data).expect("Failed to load module");

        let original_rate = edited_module.sample_rate(SAMPLE_INDEX);
        let sample_channels = edited_module.sample_channels(SAMPLE_INDEX);
        let original_data = edited_module
            .read_sample_data(SAMPLE_INDEX)
            .expect("Should read sample 33");
        let resampled = remaster::resample_audio(
            &original_data,
            original_rate as u32,
            48_000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Should resample sample 33 to 48kHz");
        let resampled_length = resampled.len() as i64 / sample_channels as i64;

        assert!(edited_module.replace_sample_data(
            SAMPLE_INDEX,
            &resampled,
            resampled_length,
            sample_channels,
            48_000,
        ));
        remaster::patch_sample_offsets(&mut edited_module, SAMPLE_INDEX, original_rate, 48_000);

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let original_path = temp_dir.path().join("original.flac");
        let edited_path = temp_dir.path().join("edited.flac");

        render_module_to_output(
            &mut original_module,
            &original_path,
            "flac",
            75,
            true,
            48_000,
            0,
        )
        .expect("Should render original module");
        render_module_to_output(
            &mut edited_module,
            &edited_path,
            "flac",
            75,
            true,
            48_000,
            0,
        )
        .expect("Should render edited module");

        let original_bytes = std::fs::read(&original_path).expect("Should read original render");
        let edited_bytes = std::fs::read(&edited_path).expect("Should read edited render");

        assert!(
            !original_bytes.is_empty(),
            "Original render should not be empty"
        );
        assert!(
            !edited_bytes.is_empty(),
            "Edited render should not be empty"
        );
        assert_ne!(
            original_bytes, edited_bytes,
            "Rendering should reflect the live edited module state"
        );
    }

    #[test]
    fn render_module_to_output_uses_shared_default_interpolation_filter() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let flac_path = temp_dir.path().join("track.flac");
        let aac_path = temp_dir.path().join("track.m4a");

        let _ = take_recorded_render_interpolation_filter();

        render_module_to_output(&mut module, &flac_path, "flac", 75, true, 48_000, 0)
            .expect("Should render FLAC with the shared default interpolation");
        assert_eq!(
            take_recorded_render_interpolation_filter(),
            Some(DEFAULT_INTERPOLATION_FILTER_LENGTH),
            "Batch FLAC rendering should use the shared interpolation default",
        );

        render_module_to_output(&mut module, &aac_path, "m4a", 75, true, 48_000, 0)
            .expect("Should render AAC with the shared default interpolation");
        assert_eq!(
            take_recorded_render_interpolation_filter(),
            Some(DEFAULT_INTERPOLATION_FILTER_LENGTH),
            "Batch AAC rendering should use the shared interpolation default",
        );
    }

    #[test]
    fn render_module_to_output_can_render_multiple_formats_from_same_live_module() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let flac_path = temp_dir.path().join("track.flac");
        let aac_path = temp_dir.path().join("track.m4a");

        render_module_to_output(&mut module, &flac_path, "flac", 75, true, 48_000, 0)
            .expect("Should render FLAC from the live module");
        render_module_to_output(&mut module, &aac_path, "m4a", 75, true, 48_000, 0)
            .expect("Should render AAC from the same live module");

        assert!(
            std::fs::metadata(&flac_path).expect("FLAC metadata").len() > 0,
            "FLAC render should write audio data",
        );
        assert!(
            std::fs::metadata(&aac_path).expect("AAC metadata").len() > 0,
            "AAC render should write audio data",
        );
    }
}
