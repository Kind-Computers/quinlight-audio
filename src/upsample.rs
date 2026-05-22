// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

//! Standalone FLAC → FLAC AI upsampling via Quinlight's spectral-consensus pipeline.
//!
//! The `upsample` CLI subcommand builds a `SampleJob` per FLAC input (treating
//! each as a non-looped one-shot buffer), runs them through all detected
//! engines using the same `RemasterEngine::remaster_samples()` pipeline the
//! tracker `convert` flow uses, and writes the Quinlight consensus result as a
//! 24-bit FLAC at 48 kHz per input (24-bit because 32-bit FLAC is still
//! considered experimental and is poorly supported by common players). The
//! per-sample cache (keyed on PCM SHA-256) is transparently shared with the
//! tracker-sample cache — same engines, same key.
//!
//! Batch mode: pass multiple files in one invocation. All jobs go into a
//! single `remaster_samples` call, so each engine's Python subprocess loads
//! its model once per batch chunk (bounded by the per-engine `max_batch_size`)
//! instead of reloading per file.
//!
//! Long inputs are split into overlapping ~20 s chunks (controlled by
//! `CHUNK_*` constants) before being handed to the engine pipeline. Each
//! chunk becomes its own `SampleJob` in the shared batch, and the 48 kHz
//! engine outputs are equal-power crossfaded back together in
//! `stitch_chunks_48k` before being written as one FLAC per input. The
//! pipeline's per-sample `target_rms` rescaling keeps levels consistent
//! across chunks without any extra work here.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use crate::batch;
use crate::openmpt::SampleLoopInfo;
use crate::remaster::{
    self, RemasterOutput, RemasterStatus, ResampleBoundaryMode, SampleJob, SampleResult,
    UpscaleMode, build_current_pipeline_reference_48k, compute_pcm_sha256, dc_free_rms,
    ensure_not_cancelled, normalize_conditioning_input, pad_for_engine, remove_dc_per_channel,
    resample_audio, sample_frame_count, scale_engine_layout, write_engine_input_wavs,
};

#[derive(Debug)]
pub enum UpsampleError {
    Fatal(String),
    QualityGate(String),
}

impl std::fmt::Display for UpsampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fatal(msg) | Self::QualityGate(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for UpsampleError {}

/// Inputs longer than this duration get split into overlapping chunks before
/// the engine pipeline sees them. Picked so a single-chunk job fits the
/// first-stage VAE encoder comfortably on 8 GB cards; on the 24 GB reference
/// card the 16-min MLK speech file OOMed at ~12 GB, which puts the safe line
/// around 30 s for that encoder.
const CHUNK_THRESHOLD_SECS: f64 = 30.0;

/// Target per-chunk duration when chunking kicks in.
const CHUNK_DURATION_SECS: f64 = 20.0;

/// Overlap between adjacent chunks, applied as an equal-power crossfade at
/// 48 kHz during stitch. Long enough to mask stochastic engine-output
/// differences at the seam, short enough that the extra compute is a
/// rounding error on full-length jobs.
const CHUNK_OVERLAP_SECS: f64 = 1.0;

/// Inclusive-exclusive frame range of one chunk at the input sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkRange {
    start: usize,
    end: usize,
}

impl ChunkRange {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// One input's plan: which batch-job indices own which chunk ranges, plus
/// everything needed to stitch the 48 kHz outputs back into a single FLAC.
struct InputPlan {
    input_path: PathBuf,
    output_path: PathBuf,
    input_rate: u32,
    input_frames: usize,
    channels: u16,
    chunks: Vec<(ChunkRange, i32)>,
}

/// Split an input buffer's frame count into overlapping chunks. Inputs at or
/// below `CHUNK_THRESHOLD_SECS` return a single span covering the whole
/// buffer. A short tail chunk (less than half a target chunk) is folded into
/// its predecessor so engines don't pay a full pass on a mostly-padded
/// fragment.
fn plan_chunks(total_frames: usize, input_rate: u32) -> Vec<ChunkRange> {
    if total_frames == 0 || input_rate == 0 {
        return Vec::new();
    }
    let total_secs = total_frames as f64 / input_rate as f64;
    if total_secs <= CHUNK_THRESHOLD_SECS {
        return vec![ChunkRange {
            start: 0,
            end: total_frames,
        }];
    }

    let chunk_frames = (CHUNK_DURATION_SECS * input_rate as f64).round() as usize;
    let overlap_frames = (CHUNK_OVERLAP_SECS * input_rate as f64).round() as usize;
    let step = chunk_frames.saturating_sub(overlap_frames).max(1);

    let mut ranges = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + chunk_frames).min(total_frames);
        ranges.push(ChunkRange { start, end });
        if end >= total_frames {
            break;
        }
        start += step;
    }

    if ranges.len() >= 2 {
        let last_len = ranges.last().unwrap().len();
        if last_len < chunk_frames / 2 {
            ranges.pop();
            ranges.last_mut().unwrap().end = total_frames;
        }
    }

    ranges
}

/// Stitch per-chunk 48 kHz engine outputs into one interleaved buffer.
/// Each chunk is placed at its input-mapped 48 kHz position and its first
/// `CHUNK_OVERLAP_SECS` worth of frames are equal-power crossfaded against
/// the previous chunk's tail. The output length is derived from the
/// original input frame count so cumulative rounding doesn't drift.
fn stitch_chunks_48k(
    chunks: &[(ChunkRange, &SampleResult)],
    input_rate: u32,
    input_frames: usize,
    channels: usize,
) -> Vec<f64> {
    if chunks.is_empty() || channels == 0 || input_rate == 0 {
        return Vec::new();
    }
    let to_48k = |n: usize| (n as u64 * 48_000u64 / input_rate as u64) as usize;
    let total_out_frames = to_48k(input_frames);
    let overlap_out_frames = (CHUNK_OVERLAP_SECS * 48_000.0).round() as usize;
    let mut out = vec![0.0f64; total_out_frames * channels];

    for (i, (range, result)) in chunks.iter().enumerate() {
        let data = &result.data;
        let chunk_out_frames = data.len() / channels;
        let pos = to_48k(range.start);
        if chunk_out_frames == 0 || pos >= total_out_frames {
            continue;
        }
        let writable = chunk_out_frames.min(total_out_frames - pos);

        if i == 0 {
            for f in 0..writable {
                for c in 0..channels {
                    out[(pos + f) * channels + c] = data[f * channels + c];
                }
            }
            continue;
        }

        let xf = overlap_out_frames.min(writable);
        for f in 0..xf {
            let t = (f as f64 + 0.5) / xf as f64;
            let (sin_t, cos_t) = (t * std::f64::consts::FRAC_PI_2).sin_cos();
            for c in 0..channels {
                let dst = (pos + f) * channels + c;
                let prev = out[dst];
                let curr = data[f * channels + c];
                out[dst] = prev * cos_t + curr * sin_t;
            }
        }
        for f in xf..writable {
            for c in 0..channels {
                out[(pos + f) * channels + c] = data[f * channels + c];
            }
        }
    }

    out
}

/// Run the upsample pipeline across one or more FLAC inputs. Each input
/// is split into one or more overlapping chunks (see `plan_chunks`); every
/// chunk becomes its own `SampleJob`, all jobs are handed to the engine
/// orchestrator in a single `remaster_samples` call so the Python
/// subprocesses load their models once per batch chunk instead of per file,
/// and the 48 kHz results are `stitch_chunks_48k`'d back into one FLAC per
/// input.
pub fn run_upsample_batch(
    inputs: &[PathBuf],
    output: Option<&Path>,
    mode: UpscaleMode,
    full_parallel: bool,
    requested_engines: &[String],
    ddim_steps: u32,
    shutdown_flag: &AtomicBool,
) -> Result<(), UpsampleError> {
    if inputs.is_empty() {
        return Err(UpsampleError::Fatal("No input files provided.".into()));
    }

    let output_paths = resolve_output_paths(inputs, output)?;

    let mut fatal_count = 0usize;
    let mut gate_count = 0usize;
    let mut sinc_fallback_count = 0usize;

    // First pass: decode each input and gate out already-48 kHz files. We do
    // this before engine detection so that a batch where every input is
    // already 48 kHz returns QualityGate cleanly even on hosts without any
    // engines installed.
    struct PreparedInput {
        input_path: PathBuf,
        output_path: PathBuf,
        data: Vec<f64>,
        rate: u32,
        channels: u16,
        bps: u32,
        stem: String,
    }

    let mut prepared: Vec<PreparedInput> = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let (data, rate, channels, bps) = match read_flac_to_f64(input) {
            Ok(tuple) => tuple,
            Err(e) => {
                eprintln!("  FATAL: read {}: {e}", input.display());
                fatal_count += 1;
                continue;
            }
        };

        if rate >= 48_000 {
            eprintln!(
                "  SKIP: {} is already at {rate} Hz (>= 48 kHz); \
                 nothing to upsample.",
                input.display(),
            );
            gate_count += 1;
            continue;
        }

        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("flac-input-{i}"));
        prepared.push(PreparedInput {
            input_path: input.clone(),
            output_path: output_paths[i].clone(),
            data,
            rate,
            channels,
            bps,
            stem,
        });
    }

    if prepared.is_empty() {
        return Err(if fatal_count > 0 {
            UpsampleError::Fatal(format!(
                "All {} input(s) failed to load before engines could run.",
                inputs.len()
            ))
        } else {
            UpsampleError::QualityGate(format!(
                "All {} input(s) are already at >= 48 kHz; nothing to upsample.",
                inputs.len()
            ))
        });
    }

    let engine = batch::detect_remaster_engine(requested_engines).map_err(UpsampleError::Fatal)?;
    let min_duration_secs = engine.min_duration_secs();

    // One tempdir shared across the batch — unique `sample_index` per job
    // prevents `sample_{i}_*.wav` filename collisions.
    let work_dir =
        tempfile::tempdir().map_err(|e| UpsampleError::Fatal(format!("tempdir: {e}")))?;

    let mut jobs: Vec<SampleJob> = Vec::with_capacity(prepared.len());
    let mut plans: Vec<InputPlan> = Vec::with_capacity(prepared.len());

    for prep in prepared {
        let PreparedInput {
            input_path,
            output_path,
            data,
            rate,
            channels,
            bps,
            stem,
        } = prep;
        let channels_usz = channels as usize;
        let input_frames = if channels_usz == 0 {
            0
        } else {
            data.len() / channels_usz
        };
        let ranges = plan_chunks(input_frames, rate);
        if ranges.is_empty() {
            eprintln!(
                "  FATAL: empty input buffer for {} (0 frames).",
                input_path.display(),
            );
            fatal_count += 1;
            continue;
        }
        let n_chunks = ranges.len();
        let duration_secs = if rate == 0 {
            0.0
        } else {
            input_frames as f64 / rate as f64
        };

        let mut chunk_entries: Vec<(ChunkRange, i32)> = Vec::with_capacity(n_chunks);
        let mut failed_chunk: Option<String> = None;
        for (k, range) in ranges.iter().enumerate() {
            let slice: Vec<f64> =
                data[(range.start * channels_usz)..(range.end * channels_usz)].to_vec();
            let index = jobs.len() as i32;
            let chunk_name = if n_chunks == 1 {
                stem.clone()
            } else {
                format!("{stem} [chunk {}/{}]", k + 1, n_chunks)
            };
            match build_sample_job_from_pcm(
                slice,
                rate,
                channels as i32,
                bps as u8,
                chunk_name,
                index,
                min_duration_secs,
                work_dir.path(),
                shutdown_flag,
            ) {
                Ok(job) => {
                    jobs.push(job);
                    chunk_entries.push((*range, index));
                }
                Err(e) => {
                    failed_chunk = Some(format!("chunk {}/{}: {e}", k + 1, n_chunks));
                    break;
                }
            }
        }

        if let Some(err) = failed_chunk {
            eprintln!("  FATAL: build job for {}: {err}", input_path.display());
            fatal_count += 1;
            // Drop any chunks already pushed for this input so the engine
            // pipeline doesn't burn time producing results we'd just discard.
            jobs.truncate(jobs.len() - chunk_entries.len());
            continue;
        }

        if n_chunks == 1 {
            eprintln!(
                "  queued {} ({} Hz, {} ch, {:.2}s)",
                input_path.display(),
                rate,
                channels,
                duration_secs,
            );
        } else {
            eprintln!(
                "  queued {} ({} Hz, {} ch, {:.2}s → {} chunks × {:.2}s w/ {:.2}s overlap)",
                input_path.display(),
                rate,
                channels,
                duration_secs,
                n_chunks,
                CHUNK_DURATION_SECS,
                CHUNK_OVERLAP_SECS,
            );
        }

        plans.push(InputPlan {
            input_path,
            output_path,
            input_rate: rate,
            input_frames,
            channels,
            chunks: chunk_entries,
        });
    }

    if jobs.is_empty() {
        return Err(if fatal_count > 0 {
            UpsampleError::Fatal(format!(
                "All {} input(s) failed to load before engines could run.",
                inputs.len()
            ))
        } else {
            UpsampleError::QualityGate(format!(
                "All {} input(s) are already at >= 48 kHz; nothing to upsample.",
                inputs.len()
            ))
        });
    }

    let total_jobs = jobs.len();
    if total_jobs == plans.len() {
        eprintln!(
            "quinlight: upsampling {} file(s) — models load once per engine per batch chunk",
            plans.len(),
        );
    } else {
        eprintln!(
            "quinlight: upsampling {} file(s) via {} chunk-jobs — models load once per engine per batch chunk",
            plans.len(),
            total_jobs,
        );
    }

    let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
    let (result_tx, result_rx) = crossbeam_channel::unbounded();
    let engine_ref = &engine;
    let mut best_by_index: std::collections::HashMap<i32, SampleResult> =
        std::collections::HashMap::new();

    std::thread::scope(|s| {
        s.spawn(move || {
            let _ = engine_ref.remaster_samples(
                jobs,
                work_dir,
                &progress_tx,
                &result_tx,
                mode,
                shutdown_flag,
                ddim_steps,
                /* progressive */ false,
                full_parallel,
            );
        });

        let progress_handle = s.spawn(move || {
            for status in progress_rx {
                match status {
                    RemasterStatus::Processing {
                        current,
                        total,
                        sample_name,
                    } => {
                        eprintln!("  [{current}/{total}] {sample_name}");
                    }
                    RemasterStatus::Log(msg) => eprintln!("  {msg}"),
                    RemasterStatus::Failed(msg) => eprintln!("  FAIL: {msg}"),
                    _ => {}
                }
            }
        });

        for output in result_rx {
            if let RemasterOutput::Final(result) = output
                && !result.data.is_empty()
            {
                // Later Final wins (mirrors `record_best_final_result` in
                // remaster.rs — the two-wave CLI path may fire Final twice).
                best_by_index.insert(result.index, result);
            }
        }

        let _ = progress_handle.join();
    });

    let mut succeeded = 0usize;
    for plan in &plans {
        // Collect SampleResults for this plan's chunks. Any missing chunk
        // means an engine didn't produce output for that segment — we can't
        // stitch without the piece, so fail the whole input.
        let mut chunk_results: Vec<(ChunkRange, &SampleResult)> =
            Vec::with_capacity(plan.chunks.len());
        let mut any_sinc = false;
        let mut engine_labels: Vec<String> = Vec::with_capacity(plan.chunks.len());
        let mut missing_chunk: Option<usize> = None;
        for (k, (range, job_idx)) in plan.chunks.iter().enumerate() {
            match best_by_index.get(job_idx) {
                Some(res) => {
                    if remaster::is_no_consensus_result(&res.engine_name) {
                        any_sinc = true;
                    }
                    if !engine_labels.contains(&res.engine_name) {
                        engine_labels.push(res.engine_name.clone());
                    }
                    chunk_results.push((*range, res));
                }
                None => {
                    missing_chunk = Some(k);
                    break;
                }
            }
        }

        if let Some(k) = missing_chunk {
            if plan.chunks.len() == 1 {
                eprintln!(
                    "  FATAL: no AI engine produced output for {}.",
                    plan.input_path.display(),
                );
            } else {
                eprintln!(
                    "  FATAL: no AI engine produced output for {} chunk {}/{}.",
                    plan.input_path.display(),
                    k + 1,
                    plan.chunks.len(),
                );
            }
            fatal_count += 1;
            continue;
        }

        let stitched = if plan.chunks.len() == 1 {
            // Single-chunk fast path: no stitching needed, clone the one result.
            chunk_results[0].1.data.clone()
        } else {
            stitch_chunks_48k(
                &chunk_results,
                plan.input_rate,
                plan.input_frames,
                plan.channels as usize,
            )
        };

        if any_sinc {
            eprintln!(
                "  QUALITY GATE: AI consensus not reached for {} ({}); writing 48 kHz SINC fallback.",
                plan.input_path.display(),
                engine_labels.join("+"),
            );
        }

        if let Some(parent) = plan.output_path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            eprintln!("  FATAL: create output dir {}: {e}", parent.display());
            fatal_count += 1;
            continue;
        }

        match write_upsample_flac_24bit(&stitched, 48_000, plan.channels, &plan.output_path) {
            Ok(()) => {
                let engine_display = if engine_labels.is_empty() {
                    "unknown".to_string()
                } else {
                    engine_labels.join("+")
                };
                if any_sinc {
                    eprintln!(
                        "  OK: {} -> {} (SINC fallback)",
                        plan.input_path.display(),
                        plan.output_path.display(),
                    );
                    sinc_fallback_count += 1;
                } else if plan.chunks.len() == 1 {
                    eprintln!(
                        "  OK: {} -> {} ({})",
                        plan.input_path.display(),
                        plan.output_path.display(),
                        engine_display,
                    );
                    succeeded += 1;
                } else {
                    eprintln!(
                        "  OK: {} -> {} ({}, {} chunks stitched)",
                        plan.input_path.display(),
                        plan.output_path.display(),
                        engine_display,
                        plan.chunks.len(),
                    );
                    succeeded += 1;
                }
            }
            Err(e) => {
                eprintln!("  FATAL: write {}: {e}", plan.output_path.display());
                fatal_count += 1;
            }
        }
    }

    eprintln!(
        "\nDone: {succeeded} AI-upsampled, {sinc_fallback_count} SINC-fallback, \
         {gate_count} skipped (>=48 kHz), {fatal_count} fatal",
    );

    if fatal_count > 0 {
        Err(UpsampleError::Fatal(format!(
            "{fatal_count} file(s) hit fatal error"
        )))
    } else if gate_count + sinc_fallback_count > 0 {
        Err(UpsampleError::QualityGate(format!(
            "{sinc_fallback_count} file(s) used 48 kHz SINC fallback (AI consensus not reached); \
             {gate_count} skipped (already >= 48 kHz)"
        )))
    } else {
        Ok(())
    }
}

/// Resolve a per-input output path vector. Policy:
/// - `None` → write next to each input with the default name.
/// - `Some(path)` that is an existing directory, or any `Some(path)` with
///   N > 1 inputs → treat as a directory and place default-named outputs inside.
/// - `Some(path)` with N == 1 and `path` not an existing dir → treat as the
///   literal single-file output path (preserves pre-batch single-file behavior).
///
/// Returns an error if two inputs resolve to the same output path.
fn resolve_output_paths(
    inputs: &[PathBuf],
    output: Option<&Path>,
) -> Result<Vec<PathBuf>, UpsampleError> {
    let paths: Vec<PathBuf> = match output {
        None => inputs
            .iter()
            .map(|p| default_upsample_output_path(p))
            .collect(),
        Some(out) => {
            let treat_as_dir = out.is_dir() || inputs.len() > 1;
            if treat_as_dir {
                if !out.is_dir() {
                    std::fs::create_dir_all(out).map_err(|e| {
                        UpsampleError::Fatal(format!("create output dir {}: {e}", out.display()))
                    })?;
                }
                inputs
                    .iter()
                    .map(|input| upsample_output_in_dir(input, out))
                    .collect()
            } else {
                vec![out.to_path_buf()]
            }
        }
    };

    let mut seen = std::collections::HashSet::new();
    for p in &paths {
        if !seen.insert(p.clone()) {
            return Err(UpsampleError::Fatal(format!(
                "Two inputs would write to the same output path: {}",
                p.display(),
            )));
        }
    }

    Ok(paths)
}

fn upsample_output_in_dir(input: &Path, dir: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "quinlight-output".into());
    dir.join(format!(
        "{stem}-Quinlight-Audio-Remastered-{}.flac",
        batch::format_rate_khz(48_000)
    ))
}

/// Decode a FLAC file to interleaved `f64` PCM in `[-1.0, 1.0]`.
/// Returns `(data, sample_rate_hz, channels, bits_per_sample)`.
/// Rejects `channels` outside `{1, 2}` — engines + ffmpeg layout only handle
/// mono/stereo.
fn read_flac_to_f64(path: &Path) -> Result<(Vec<f64>, u32, u16, u32), String> {
    use flac_codec::decode::{FlacSampleReader, Metadata};

    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read FLAC file {}: {e}", path.display()))?;
    let reader = FlacSampleReader::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("FLAC decoder init for {}: {e:?}", path.display()))?;

    let rate = reader.sample_rate();
    let channels = reader.channel_count();
    let bps = reader.bits_per_sample();

    if !(1..=2).contains(&channels) {
        return Err(format!(
            "Unsupported channel count {channels} in {}; quinlight upsample accepts mono or stereo only.",
            path.display(),
        ));
    }
    if bps == 0 || bps > 32 {
        return Err(format!(
            "Unsupported bit depth {bps} in {}.",
            path.display()
        ));
    }

    // Sample iterator yields sign-extended i32 regardless of source bit depth.
    let i32_samples: Vec<i32> = reader
        .into_iter()
        .collect::<Result<_, _>>()
        .map_err(|e| format!("FLAC decode error for {}: {e:?}", path.display()))?;

    let scale = ((1i64) << (bps - 1)) as f64;
    let data: Vec<f64> = i32_samples.iter().map(|&s| s as f64 / scale).collect();

    Ok((data, rate, channels as u16, bps))
}

/// Build a single non-looped `SampleJob` from an arbitrary PCM buffer. Mirrors
/// the per-sample body of `extract_sample_jobs` (`src/remaster.rs:7157`) but
/// skips the tracker-side `OriginalSample` input.
///
/// `sample_index` is threaded through to `write_engine_input_wavs` so a batch
/// of jobs sharing one work directory gets distinct `sample_{i}_*.wav` names.
#[allow(clippy::too_many_arguments)]
fn build_sample_job_from_pcm(
    mut original_data: Vec<f64>,
    input_rate: u32,
    channels: i32,
    bits_per_sample: u8,
    name: String,
    sample_index: i32,
    min_duration_secs: f64,
    work_dir: &Path,
    cancel_flag: &AtomicBool,
) -> Result<SampleJob, String> {
    if !(1..=2).contains(&channels) {
        return Err(format!(
            "Unsupported channel count {channels}; mono or stereo only.",
        ));
    }
    let channels_usz = channels as usize;

    remove_dc_per_channel(&mut original_data, channels_usz);

    let source_length_frames = sample_frame_count(&original_data, channels_usz);
    let resampled_48k =
        build_current_pipeline_reference_48k(&original_data, input_rate, channels_usz)?;
    let original_length_48k_frames = sample_frame_count(&resampled_48k, channels_usz);

    let native_rate = input_rate;
    let cond_rate: u32 = if native_rate < 24_000 { 24_000 } else { 48_000 };

    let native_min_samples =
        (min_duration_secs * native_rate as f64).ceil() as usize * channels_usz;
    let (mut native_padded, native_layout) = pad_for_engine(
        &original_data,
        channels_usz,
        native_min_samples,
        SampleLoopInfo::none(),
    );
    normalize_conditioning_input(&mut native_padded, channels_usz);
    let target_rms = dc_free_rms(&original_data, channels_usz);

    ensure_not_cancelled(cancel_flag)?;

    let native_inputs_all = write_engine_input_wavs(
        &native_padded,
        channels,
        sample_index,
        work_dir,
        native_rate,
        "_native",
    )?;

    ensure_not_cancelled(cancel_flag)?;

    let (
        conditioning_inputs,
        engine_input_layout,
        native_inputs,
        native_rate_hz,
        native_input_layout,
    ) = if native_rate == cond_rate {
        (native_inputs_all, native_layout, Vec::new(), 0u32, None)
    } else {
        let mut cond_data = resample_audio(
            &native_padded,
            native_rate,
            cond_rate,
            channels_usz,
            ResampleBoundaryMode::OneShot,
        )?;
        normalize_conditioning_input(&mut cond_data, channels_usz);
        let cond_inputs =
            write_engine_input_wavs(&cond_data, channels, sample_index, work_dir, cond_rate, "")?;
        let cond_layout = scale_engine_layout(native_layout, native_rate, cond_rate);
        (
            cond_inputs,
            cond_layout,
            native_inputs_all,
            native_rate,
            native_layout,
        )
    };

    let pcm_sha256 = compute_pcm_sha256(&resampled_48k);

    Ok(SampleJob {
        index: sample_index,
        name,
        original_data,
        rate: input_rate as i32,
        output_sample_rate_hz: 48_000,
        channels,
        bits_per_sample,
        source_length_frames,
        looped: false,
        loop_info: SampleLoopInfo::none(),
        conditioning_inputs,
        conditioning_rate_hz: cond_rate,
        pcm_sha256,
        target_rms,
        reference_48k: resampled_48k,
        original_length_48k_frames,
        engine_input_layout,
        native_inputs,
        native_rate_hz,
        native_input_layout,
    })
}

fn default_upsample_output_path(input: &Path) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "quinlight-output".into());
    parent.join(format!(
        "{stem}-Quinlight-Audio-Remastered-{}.flac",
        batch::format_rate_khz(48_000)
    ))
}

/// Write `data` to a 24-bit FLAC. If the peak magnitude exceeds 1.0, scale
/// the entire buffer by `1/peak` before quantizing so the max lands at
/// full-scale without clipping. Never clamps.
///
/// 24-bit (not 32-bit) is used for CLI-facing output because 32-bit FLAC is
/// still considered experimental and is poorly supported by common players
/// and DAWs. 24-bit is the safe, widely-supported default. (The internal
/// sample cache in `src/remaster.rs` stays at 32-bit — via 32-bit FLAC
/// (`build_flac_i32`) and 32-bit WAV-in-zstd — since those files are never
/// opened by external tools.)
fn write_upsample_flac_24bit(
    data: &[f64],
    sample_rate: u32,
    channels: u16,
    path: &Path,
) -> Result<(), String> {
    use flac_codec::encode::{FlacSampleWriter, Options};

    let peak = data.iter().fold(0.0_f64, |a, &s| a.max(s.abs()));
    let gain = if peak > 1.0 { 1.0 / peak } else { 1.0 };
    if peak > 1.0 {
        eprintln!("  peak {peak:.6} > 1.0; scaled by {gain:.6} to avoid clipping.");
    }

    // Signed 24-bit PCM range is [-2^23, 2^23 - 1].
    const MAX_24: i32 = (1i32 << 23) - 1;
    const MIN_24: i32 = -(1i32 << 23);
    let scale = MAX_24 as f64;
    let i32_samples: Vec<i32> = data
        .iter()
        .map(|&s| {
            let v = (s * gain * scale).round();
            if v >= MAX_24 as f64 {
                MAX_24
            } else if v <= MIN_24 as f64 {
                MIN_24
            } else {
                v as i32
            }
        })
        .collect();

    let tmp_path = tmp_path_for(path);
    {
        let mut file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("create {}: {e}", tmp_path.display()))?;
        let mut writer = FlacSampleWriter::new(
            &mut file,
            Options::default(),
            sample_rate,
            24,
            channels as u8,
            Some(data.len() as u64),
        )
        .map_err(|e| format!("FLAC encoder init: {e:?}"))?;
        writer
            .write(&i32_samples)
            .map_err(|e| format!("FLAC encode: {e:?}"))?;
        writer
            .finalize()
            .map_err(|e| format!("FLAC finalize: {e:?}"))?;
    }
    std::fs::rename(&tmp_path, path)
        .map_err(|e| format!("rename {} → {}: {e}", tmp_path.display(), path.display()))?;

    Ok(())
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|n| format!(".{}.tmp", n.to_string_lossy()))
        .unwrap_or_else(|| ".quinlight.tmp".into());
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(file_name),
        _ => PathBuf::from(file_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn write_test_flac(path: &Path, data: &[i32], rate: u32, channels: u8, bps: u32) {
        use flac_codec::encode::{FlacSampleWriter, Options};
        let mut file = std::fs::File::create(path).expect("create test flac");
        let mut writer = FlacSampleWriter::new(
            &mut file,
            Options::default(),
            rate,
            bps,
            channels,
            Some(data.len() as u64),
        )
        .expect("flac init");
        writer.write(data).expect("flac write");
        writer.finalize().expect("flac finalize");
    }

    #[test]
    fn read_flac_to_f64_roundtrips_32bit_mono() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sine.flac");
        let rate = 44_100u32;
        let channels = 1u8;
        let bps = 32u32;
        // 4096-sample 1 kHz sine, amplitude 0.5.
        let orig: Vec<f64> = (0..4096)
            .map(|n| 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / rate as f64).sin())
            .collect();
        let scale = ((1i64) << (bps - 1)) as f64;
        let i32s: Vec<i32> = orig.iter().map(|&s| (s * scale).round() as i32).collect();
        write_test_flac(&path, &i32s, rate, channels, bps);

        let (got, got_rate, got_ch, got_bps) = read_flac_to_f64(&path).unwrap();
        assert_eq!(got_rate, rate);
        assert_eq!(got_ch, 1);
        assert_eq!(got_bps, bps);
        assert_eq!(got.len(), orig.len());
        let max_err = got
            .iter()
            .zip(orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-9, "max_err {max_err}");
    }

    #[test]
    fn read_flac_to_f64_rejects_five_channel_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("surround.flac");
        let rate = 48_000u32;
        let channels = 5u8;
        let bps = 16u32;
        let frames = 256;
        let samples: Vec<i32> = vec![0; frames * channels as usize];
        write_test_flac(&path, &samples, rate, channels, bps);
        let err = read_flac_to_f64(&path).unwrap_err();
        assert!(err.contains("Unsupported channel count 5"), "err: {err}");
    }

    #[test]
    fn build_sample_job_from_pcm_22050_mono_uses_24khz_conditioning_and_native_wavs() {
        let dir = tempfile::tempdir().unwrap();
        let rate = 22_050u32;
        let channels = 1i32;
        // 1 second of sine — far shorter than 5.12s, so pad_for_engine should
        // produce a Repeated (non-looped tile) layout.
        let frames = rate as usize;
        let data: Vec<f64> = (0..frames)
            .map(|n| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * n as f64 / rate as f64).sin())
            .collect();
        let flag = AtomicBool::new(false);
        let job = build_sample_job_from_pcm(
            data,
            rate,
            channels,
            32,
            "test".into(),
            0,
            5.12,
            dir.path(),
            &flag,
        )
        .unwrap();

        assert_eq!(job.index, 0);
        assert_eq!(job.rate, rate as i32);
        assert_eq!(job.output_sample_rate_hz, 48_000);
        assert_eq!(job.channels, 1);
        assert!(!job.looped);
        assert_eq!(job.conditioning_rate_hz, 24_000);
        assert_eq!(job.native_rate_hz, rate);
        assert!(!job.native_inputs.is_empty());
        assert!(!job.conditioning_inputs.is_empty());
        assert_eq!(job.source_length_frames, frames as i64);

        // ~1s at 22050 in → ~48000 frames at 48k.
        let expected_48k = (frames as f64 * 48_000.0 / rate as f64).round() as i64;
        let diff = (job.original_length_48k_frames - expected_48k).abs();
        assert!(
            diff <= 2,
            "48k frame count {}, expected ~{expected_48k}",
            job.original_length_48k_frames
        );

        for input in job
            .conditioning_inputs
            .iter()
            .chain(job.native_inputs.iter())
        {
            assert!(input.input_path.exists(), "missing {:?}", input.input_path);
        }
    }

    #[test]
    fn batch_job_indices_namespace_engine_input_wavs() {
        // Two jobs sharing one work dir must get distinct sample_{i}_*.wav
        // filenames — otherwise the second job overwrites the first.
        let dir = tempfile::tempdir().unwrap();
        let rate = 22_050u32;
        let frames = rate as usize;
        let data0: Vec<f64> = (0..frames)
            .map(|n| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * n as f64 / rate as f64).sin())
            .collect();
        let data1: Vec<f64> = (0..frames)
            .map(|n| 0.25 * (2.0 * std::f64::consts::PI * 220.0 * n as f64 / rate as f64).sin())
            .collect();
        let flag = AtomicBool::new(false);
        let job0 =
            build_sample_job_from_pcm(data0, rate, 1, 32, "a".into(), 0, 5.12, dir.path(), &flag)
                .unwrap();
        let job1 =
            build_sample_job_from_pcm(data1, rate, 1, 32, "b".into(), 1, 5.12, dir.path(), &flag)
                .unwrap();

        assert_eq!(job0.index, 0);
        assert_eq!(job1.index, 1);

        let paths0: Vec<_> = job0
            .conditioning_inputs
            .iter()
            .chain(job0.native_inputs.iter())
            .map(|p| p.input_path.clone())
            .collect();
        let paths1: Vec<_> = job1
            .conditioning_inputs
            .iter()
            .chain(job1.native_inputs.iter())
            .map(|p| p.input_path.clone())
            .collect();

        for p in &paths0 {
            assert!(p.exists(), "missing job0 input {p:?}");
            assert!(!paths1.contains(p), "job1 collided with job0 at {p:?}");
        }
        for p in &paths1 {
            assert!(p.exists(), "missing job1 input {p:?}");
        }
    }

    #[test]
    fn default_output_path_follows_naming_convention() {
        assert_eq!(
            default_upsample_output_path(Path::new("/tmp/track.flac")),
            PathBuf::from("/tmp/track-Quinlight-Audio-Remastered-48Khz.flac"),
        );
        assert_eq!(
            default_upsample_output_path(Path::new("song.flac")),
            PathBuf::from("song-Quinlight-Audio-Remastered-48Khz.flac"),
        );
    }

    #[test]
    fn tmp_path_for_preserves_parent_and_prefixes_dot() {
        assert_eq!(
            tmp_path_for(Path::new("/a/b/out.flac")),
            PathBuf::from("/a/b/.out.flac.tmp"),
        );
        assert_eq!(
            tmp_path_for(Path::new("out.flac")),
            PathBuf::from(".out.flac.tmp")
        );
    }

    #[test]
    fn run_upsample_batch_rejects_already_48khz_input() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hi.flac");
        let rate = 48_000u32;
        let frames = 256;
        let samples: Vec<i32> = vec![0; frames];
        write_test_flac(&path, &samples, rate, 1, 16);

        let flag = AtomicBool::new(false);
        let err = run_upsample_batch(
            std::slice::from_ref(&path),
            None,
            UpscaleMode::CpuOnly,
            false,
            &[],
            25,
            &flag,
        )
        .expect_err("48 kHz input should be rejected");
        match err {
            UpsampleError::QualityGate(msg) => {
                // All inputs ≥ 48 kHz hits the pre-engine QualityGate branch.
                assert!(msg.contains(">= 48 kHz"), "{msg}");
            }
            _ => panic!("expected QualityGate, got {err:?}"),
        }
    }

    #[test]
    fn resolve_output_paths_single_input_no_output_uses_default() {
        let input = PathBuf::from("/tmp/song.flac");
        let got = resolve_output_paths(std::slice::from_ref(&input), None).unwrap();
        assert_eq!(
            got,
            vec![PathBuf::from(
                "/tmp/song-Quinlight-Audio-Remastered-48Khz.flac"
            )]
        );
    }

    #[test]
    fn resolve_output_paths_single_input_with_file_output_uses_literal_path() {
        let input = PathBuf::from("/tmp/song.flac");
        let out = PathBuf::from("/tmp/custom-name.flac");
        let got = resolve_output_paths(std::slice::from_ref(&input), Some(&out)).unwrap();
        assert_eq!(got, vec![out]);
    }

    #[test]
    fn resolve_output_paths_multiple_inputs_write_into_output_dir() {
        let out_dir = tempfile::tempdir().unwrap();
        let inputs = vec![
            PathBuf::from("/tmp/a.flac"),
            PathBuf::from("/elsewhere/b.flac"),
        ];
        let got = resolve_output_paths(&inputs, Some(out_dir.path())).unwrap();
        assert_eq!(
            got,
            vec![
                out_dir
                    .path()
                    .join("a-Quinlight-Audio-Remastered-48Khz.flac"),
                out_dir
                    .path()
                    .join("b-Quinlight-Audio-Remastered-48Khz.flac"),
            ]
        );
    }

    #[test]
    fn resolve_output_paths_multiple_inputs_creates_missing_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let out_dir = tmp.path().join("nested").join("deep");
        assert!(!out_dir.exists());
        let inputs = vec![PathBuf::from("/tmp/a.flac"), PathBuf::from("/tmp/b.flac")];
        let _ = resolve_output_paths(&inputs, Some(&out_dir)).unwrap();
        assert!(out_dir.is_dir(), "output dir should have been created");
    }

    #[test]
    fn resolve_output_paths_rejects_colliding_outputs() {
        // Two inputs with identical stems in different dirs collide under a shared output dir.
        let out_dir = tempfile::tempdir().unwrap();
        let inputs = vec![
            PathBuf::from("/one/song.flac"),
            PathBuf::from("/two/song.flac"),
        ];
        let err = resolve_output_paths(&inputs, Some(out_dir.path())).expect_err("must collide");
        match err {
            UpsampleError::Fatal(msg) => {
                assert!(msg.contains("same output path"), "{msg}");
                assert!(
                    msg.contains("song-Quinlight-Audio-Remastered-48Khz.flac"),
                    "{msg}"
                );
            }
            _ => panic!("expected Fatal, got {err:?}"),
        }
    }

    #[test]
    fn run_upsample_batch_with_empty_input_list_is_fatal() {
        let flag = AtomicBool::new(false);
        let err = run_upsample_batch(&[], None, UpscaleMode::CpuOnly, false, &[], 25, &flag)
            .expect_err("empty input list should be rejected");
        assert!(matches!(err, UpsampleError::Fatal(_)), "got {err:?}");
    }

    #[test]
    fn write_upsample_flac_24bit_emits_24bit_file_and_never_clips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.flac");
        // Peak > 1.0 — writer must scale (not clip) so the output peak sits at full scale.
        let data: Vec<f64> = vec![0.0, 0.5, 1.5, -1.2, 0.0];
        write_upsample_flac_24bit(&data, 48_000, 1, &path).unwrap();
        assert!(path.exists());
        let (got, _, _, got_bps) = read_flac_to_f64(&path).unwrap();
        assert_eq!(
            got_bps, 24,
            "CLI output must be 24-bit (32-bit is experimental)"
        );
        let max_abs = got.iter().fold(0.0_f64, |a, &s| a.max(s.abs()));
        assert!(
            (max_abs - 1.0).abs() < 1e-6,
            "expected post-write peak ≈ 1.0, got {max_abs}",
        );
    }

    #[test]
    fn write_upsample_flac_24bit_round_trips_audio_within_quantization_floor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sine.flac");
        let rate = 48_000u32;
        let frames = 4096;
        let data: Vec<f64> = (0..frames)
            .map(|n| 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / rate as f64).sin())
            .collect();
        write_upsample_flac_24bit(&data, rate, 1, &path).unwrap();
        let (got, got_rate, got_ch, got_bps) = read_flac_to_f64(&path).unwrap();
        assert_eq!(got_bps, 24);
        assert_eq!(got_rate, rate);
        assert_eq!(got_ch, 1);
        assert_eq!(got.len(), frames);
        // 24-bit quantization noise floor is ~2^-23 ≈ 1.2e-7.
        let max_err = got
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 2e-7, "24-bit round-trip max_err {max_err}");
    }

    fn fake_result(index: i32, data: Vec<f64>, channels: i32) -> SampleResult {
        let length_frames = (data.len() / channels.max(1) as usize) as i64;
        SampleResult {
            index,
            data,
            length_frames,
            channels,
            sample_rate_hz: 48_000,
            engine_name: "TestEngine".into(),
            discovered_loops: None,
        }
    }

    #[test]
    fn plan_chunks_short_input_returns_single_span() {
        // 10 s at 22050 — well below the 30 s threshold.
        let frames = 22_050 * 10;
        let ranges = plan_chunks(frames, 22_050);
        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0],
            ChunkRange {
                start: 0,
                end: frames
            }
        );
    }

    #[test]
    fn plan_chunks_at_threshold_stays_single_span() {
        // Boundary: exactly the threshold must not trigger chunking.
        let frames = (CHUNK_THRESHOLD_SECS * 22_050.0) as usize;
        let ranges = plan_chunks(frames, 22_050);
        assert_eq!(ranges.len(), 1);
    }

    #[test]
    fn plan_chunks_long_input_has_correct_overlap() {
        // 90 s at 22050 → multiple 20 s chunks stepping by 19 s.
        let rate = 22_050u32;
        let frames = rate as usize * 90;
        let ranges = plan_chunks(frames, rate);
        assert!(
            ranges.len() >= 4,
            "expected >=4 chunks, got {}",
            ranges.len()
        );

        // First chunk starts at 0.
        assert_eq!(ranges[0].start, 0);
        // Last chunk ends at total_frames.
        assert_eq!(ranges.last().unwrap().end, frames);

        // Adjacent chunks overlap by CHUNK_OVERLAP_SECS worth of frames.
        let overlap = (CHUNK_OVERLAP_SECS * rate as f64).round() as usize;
        for pair in ranges.windows(2) {
            // Tail merge may have widened the last chunk, so only check
            // pairs where the left chunk is a regular-sized one.
            let left_len = pair[0].len();
            let chunk = (CHUNK_DURATION_SECS * rate as f64).round() as usize;
            if left_len == chunk {
                assert_eq!(pair[0].end - pair[1].start, overlap);
            }
        }
    }

    #[test]
    fn plan_chunks_tail_shorter_than_half_merges_into_previous() {
        // 41 s at 22050: chunk=20s, step=19s → naive plan is 3 chunks
        // ([0,20s) [19s,39s) [38s,41s)). The 3 s tail is below chunk/2=10s,
        // so it must be folded into the prior chunk, leaving 2 chunks total.
        let rate = 22_050u32;
        let frames = rate as usize * 41;
        let ranges = plan_chunks(frames, rate);

        assert_eq!(
            ranges.len(),
            2,
            "tiny tail should have merged, got {} chunks",
            ranges.len(),
        );
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[1].end, frames, "tail must extend to total_frames");

        let chunk = (CHUNK_DURATION_SECS * rate as f64).round() as usize;
        assert!(
            ranges[1].len() > chunk,
            "merged chunk should be widened, got {} (chunk {})",
            ranges[1].len(),
            chunk,
        );
    }

    #[test]
    fn plan_chunks_zero_frames_returns_empty() {
        assert!(plan_chunks(0, 48_000).is_empty());
    }

    #[test]
    fn stitch_chunks_48k_single_chunk_copies_data() {
        // 2 s at 48 kHz — chunk covers the whole 1 s input at 24 kHz.
        let input_rate = 24_000u32;
        let input_frames = 24_000; // 1 s
        let channels = 1;
        let mut data = vec![0.0f64; 48_000]; // 1 s at 48 kHz
        for (i, s) in data.iter_mut().enumerate() {
            *s = (i as f64 / 48_000.0).sin();
        }
        let result = fake_result(0, data.clone(), 1);
        let chunks = vec![(
            ChunkRange {
                start: 0,
                end: input_frames,
            },
            &result,
        )];
        let out = stitch_chunks_48k(&chunks, input_rate, input_frames, channels);
        assert_eq!(out.len(), 48_000);
        assert_eq!(out, data);
    }

    #[test]
    fn stitch_chunks_48k_overlap_preserves_constant_value() {
        // Two chunks of constant value 1.0 with overlap — equal-power
        // crossfade should hold the output at 1.0 throughout the overlap
        // (cos + sin at matched phases sums to √2 * (cos+sin)/√2 = 1 when
        // both sources carry the same signal).
        let input_rate = 22_050u32;
        // Two 20 s chunks with 1 s overlap → 39 s total input.
        let first_len = 22_050 * 20;
        let overlap_in = 22_050;
        let second_start = first_len - overlap_in;
        let total_frames = second_start + first_len; // = 22050 * 39

        let out_rate = 48_000u64;
        let to_48k = |n: usize| (n as u64 * out_rate / input_rate as u64) as usize;
        let first_out_len = to_48k(first_len);
        let second_out_len = to_48k(first_len);

        let data = vec![1.0f64; first_out_len];
        let r0 = fake_result(0, data.clone(), 1);
        let r1 = fake_result(1, vec![1.0f64; second_out_len], 1);

        let chunks = vec![
            (
                ChunkRange {
                    start: 0,
                    end: first_len,
                },
                &r0,
            ),
            (
                ChunkRange {
                    start: second_start,
                    end: total_frames,
                },
                &r1,
            ),
        ];
        let out = stitch_chunks_48k(&chunks, input_rate, total_frames, 1);

        // Every sample that ends up covered should sit right near 1.0. The
        // equal-power cross-sum of two identical signals is cos + sin ≤ √2,
        // but critically >= 1.0 at all points, so set a tight lower bound.
        let xf_out = (CHUNK_OVERLAP_SECS * 48_000.0) as usize;
        let xf_start = to_48k(second_start);
        for (i, &s) in out.iter().enumerate() {
            if i < xf_start {
                assert!((s - 1.0).abs() < 1e-12, "pre-overlap sample {i}: {s}");
            } else if i < xf_start + xf_out {
                // Equal-power sum of two unit sources: cos(θ) + sin(θ), peak √2 ≈ 1.414.
                assert!(
                    (s - 1.0).abs() <= (2.0_f64.sqrt() - 1.0) + 1e-9,
                    "overlap sample {i}: {s}"
                );
                assert!(s >= 1.0 - 1e-9, "overlap sample dropped below 1.0: {s}");
            }
        }
    }

    #[test]
    fn stitch_chunks_48k_output_length_matches_input_mapped_to_48k() {
        // 39 s input at 22050 → 39 s output at 48000 = 48000 * 39 = 1_872_000.
        let input_rate = 22_050u32;
        let first_len = 22_050 * 20;
        let overlap_in = 22_050;
        let second_start = first_len - overlap_in;
        let total_frames = second_start + first_len;
        let to_48k = |n: usize| (n as u64 * 48_000u64 / input_rate as u64) as usize;
        let expected_out = to_48k(total_frames);

        let r0 = fake_result(0, vec![0.5f64; to_48k(first_len)], 1);
        let r1 = fake_result(1, vec![0.5f64; to_48k(first_len)], 1);

        let chunks = vec![
            (
                ChunkRange {
                    start: 0,
                    end: first_len,
                },
                &r0,
            ),
            (
                ChunkRange {
                    start: second_start,
                    end: total_frames,
                },
                &r1,
            ),
        ];
        let out = stitch_chunks_48k(&chunks, input_rate, total_frames, 1);
        assert_eq!(out.len(), expected_out);
    }

    #[test]
    fn stitch_chunks_48k_handles_stereo_interleaved() {
        // Verify channel interleaving is preserved: L=0.2, R=-0.4 everywhere.
        let input_rate = 24_000u32;
        let input_frames = 48_000; // 2 s
        let channels = 2usize;
        let out_frames = 96_000usize; // 2 s at 48k
        let mut data = Vec::with_capacity(out_frames * channels);
        for _ in 0..out_frames {
            data.push(0.2);
            data.push(-0.4);
        }
        let result = fake_result(0, data, 2);
        let chunks = vec![(
            ChunkRange {
                start: 0,
                end: input_frames,
            },
            &result,
        )];
        let out = stitch_chunks_48k(&chunks, input_rate, input_frames, channels);
        assert_eq!(out.len(), out_frames * channels);
        for frame in 0..out_frames {
            assert!(
                (out[frame * 2] - 0.2).abs() < 1e-12,
                "L drifted at frame {frame}"
            );
            assert!(
                (out[frame * 2 + 1] - -0.4).abs() < 1e-12,
                "R drifted at frame {frame}"
            );
        }
    }
}
