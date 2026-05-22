// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(
    clippy::approx_constant,
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::needless_lifetimes,
    clippy::ptr_arg,
    clippy::question_mark,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::unnecessary_min_or_max
)]

use crossbeam_channel::Sender;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use ac_ffmpeg::{
    codec::audio::{
        AudioFrame, AudioFrameMut, AudioResampler,
        frame::{
            ChannelLayout, SampleFormat as FfmpegSampleFormat, get_channel_layout,
            get_sample_format,
        },
    },
    time::{TimeBase, Timestamp},
};

static NO_CACHE: AtomicBool = AtomicBool::new(false);

/// Disable sample cache reads and writes (for debugging).
pub fn set_no_cache(v: bool) {
    NO_CACHE.store(v, Ordering::Relaxed);
}

pub use crate::cleanup::{CleanupEngineVersion, CleanupMode, CleanupSettings};
use crate::cleanup::{RetiredCleanupPreset, apply_cleanup, apply_retired_cleanup_preset};
use crate::engine::{self, UpsampleEngine};
use crate::openmpt::{Module, SampleLoopInfo, SampleLoopMode, SampleLoopRegion};
use crate::simd;

/// Set a child process to the given CFS nice level so AI resampling
/// doesn't starve the GUI and audio threads. Callers pass a staggered
/// value (e.g., 15 + engine_index) so concurrent engines get asymmetric
/// scheduling weight — the kernel tends to keep the favored one on CPU
/// longer per time slice, reducing migrations and cache thrash.
fn nice_child(child: &std::process::Child, nice: i32) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        unsafe {
            libc::setpriority(libc::PRIO_PROCESS as u32, pid as u32, nice);
        }
    }
}

pub(crate) const REMASTER_CANCELLED_ERROR: &str = "Cancelled";

pub(crate) fn is_cancelled_error(error: &str) -> bool {
    error == REMASTER_CANCELLED_ERROR
}

fn cancelled_error() -> String {
    REMASTER_CANCELLED_ERROR.to_string()
}

fn cancellation_requested(cancel_flag: &AtomicBool) -> bool {
    cancel_flag.load(Ordering::Relaxed)
}

pub(crate) fn ensure_not_cancelled(cancel_flag: &AtomicBool) -> Result<(), String> {
    if cancellation_requested(cancel_flag) {
        Err(cancelled_error())
    } else {
        Ok(())
    }
}

fn kill_child_best_effort(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpscaleMode {
    /// 1 GPU batch + 1 CPU batch with dynamic GPU acquisition
    Hybrid,
    /// 2 CPU workers, no GPU (default when no GPU is detected)
    CpuOnly,
    /// 1 GPU worker, no CPU (default when a GPU is detected)
    GpuOnly,
}

/// One-line shell command users paste to install all engines. The installer
/// script handles apt deps, GPU-specific torch wheels, venv setup, every AI
/// engine, and weight downloads in one shot.
pub const INSTALL_COMMAND: &str = "./install_prerequisites.sh";

/// URL to the installer script in the upstream repo, for users who installed
/// from a binary release and don't have the source checked out.
pub const INSTALL_SCRIPT_URL: &str = "https://raw.githubusercontent.com/Kind-Computers/quinlight-audio/main/install_prerequisites.sh";

#[derive(Debug, Clone)]
pub enum RemasterStatus {
    /// No AI remastering engines detected
    Unavailable,
    /// Ready to remaster
    Ready,
    /// Processing sample N of M
    Processing {
        current: i32,
        total: i32,
        sample_name: String,
    },
    /// Cancellation requested; worker is shutting down.
    Cancelling,
    /// Persistent log line (accumulates in GUI, not overwritten like Processing)
    Log(String),
    /// One selected engine finished a specific sample.
    EngineProgress {
        sample_index: i32,
        engines_done: i32,
        engines_total: i32,
    },
    /// All done
    Complete,
    /// Worker stopped due to cancellation.
    Cancelled,
    /// Failed with error
    Failed(String),
}

pub struct RemasterEngine {
    engines: Vec<Box<dyn UpsampleEngine>>,
    /// Installed engines the user has toggled off. Only invoked when a
    /// primary engine scores below `QUINLIGHT_USABLE_SCORE_FLOOR` for a
    /// sample (see `remaster_samples` fallback wave).
    fallback_engines: Vec<Box<dyn UpsampleEngine>>,
}

impl RemasterEngine {
    pub fn empty() -> Self {
        RemasterEngine {
            engines: Vec::new(),
            fallback_engines: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_engines(engines: Vec<Box<dyn UpsampleEngine>>) -> Self {
        RemasterEngine {
            engines,
            fallback_engines: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_engines_with_fallback(
        engines: Vec<Box<dyn UpsampleEngine>>,
        fallback_engines: Vec<Box<dyn UpsampleEngine>>,
    ) -> Self {
        RemasterEngine {
            engines,
            fallback_engines,
        }
    }

    pub fn is_available(&self) -> bool {
        !self.engines.is_empty()
    }

    pub fn engine_name(&self) -> String {
        if self.is_available() {
            QUINLIGHT_NAME.into()
        } else {
            "none".into()
        }
    }

    #[allow(dead_code)]
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    #[allow(dead_code)]
    pub(crate) fn eligible_enabled_engine_count_for_rate(
        &self,
        original_rate_hz: i32,
        enabled: &[String],
    ) -> i32 {
        let original_rate_hz = original_rate_hz.max(0) as u32;
        self.engines
            .iter()
            .filter(|engine| {
                enabled
                    .iter()
                    .any(|name| name.eq_ignore_ascii_case(engine.name()))
                    && engine.supports_original_rate(original_rate_hz)
            })
            .count() as i32
    }

    pub fn min_duration_secs(&self) -> f64 {
        self.engines
            .iter()
            .map(|e| e.min_duration_secs())
            .fold(0.0f64, f64::max)
            .max(5.12) // fallback if no engines
    }

    pub fn available_engine_names(&self) -> Vec<&str> {
        self.engines.iter().map(|e| e.name()).collect()
    }

    /// Names of engines that are NOT currently detected. Returns None when
    /// every supported engine is installed.
    pub fn missing_engine_names(&self) -> Option<Vec<&'static str>> {
        let detected: Vec<&str> = self.engines.iter().map(|e| e.name()).collect();
        let mut missing = Vec::new();
        for name in ["AudioSR", "LavaSR", "FLowHigh", "AP-BWE"] {
            if !detected.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                missing.push(name);
            }
        }
        if missing.is_empty() {
            None
        } else {
            Some(missing)
        }
    }
}

/// User-facing install instructions. The installer script is idempotent, so we
/// point at it whether the venv is empty or only partially populated.
pub fn install_instructions() -> String {
    format!(
        "Run {INSTALL_COMMAND} from the Quinlight source repo.\n\
         \n\
         It installs apt deps, the right torch wheel for your GPU\n\
         (CUDA / ROCm / XPU / CPU), all four AI engines\n\
         (AudioSR, LavaSR, FLowHigh, AP-BWE), and their weights.\n\
         \n\
         If you don't have the source, download the script from:\n\
         {INSTALL_SCRIPT_URL}"
    )
}

impl RemasterEngine {
    /// Detect all available upsampling engines.
    pub fn detect() -> Self {
        RemasterEngine {
            engines: engine::detect_engines(),
            fallback_engines: Vec::new(),
        }
    }

    /// Detect engines and partition into primary (names in `enabled`) and
    /// fallback (installed but not in `enabled`). The fallback set is run
    /// automatically for any sample where a primary engine scores below
    /// `QUINLIGHT_USABLE_SCORE_FLOOR`, strengthening consensus on the samples
    /// that need it without burning compute on samples where primaries pass.
    pub fn detect_with_fallback(enabled: &[String]) -> Self {
        let all = engine::detect_engines();
        let (engines, fallback_engines): (Vec<_>, Vec<_>) = all.into_iter().partition(|e| {
            enabled
                .iter()
                .any(|name| name.eq_ignore_ascii_case(e.name()))
        });
        RemasterEngine {
            engines,
            fallback_engines,
        }
    }

    /// Remaster samples using all selected AI engines.
    /// Raw per-engine candidates stream immediately. When `progressive` is
    /// true (GUI), a Quinlight `Final` is emitted as soon as 2 engines have
    /// produced candidates per sample and again if a 3rd candidate refines
    /// the set. When `progressive` is false (CLI), `Final` is emitted exactly
    /// once per sample after all eligible engines complete, so the spectral
    /// intersection runs only once per sample.
    ///
    /// Two-wave architecture: the primary wave runs `self.engines` (enabled by
    /// the user). For any sample where at least one primary engine scored
    /// below `QUINLIGHT_USABLE_SCORE_FLOOR`, a fallback wave runs
    /// `self.fallback_engines` (installed but disabled) for only those
    /// samples to strengthen consensus. Samples whose primaries all passed
    /// take the fast path with no extra compute.
    pub fn remaster_samples(
        &self,
        jobs: Vec<SampleJob>,
        work_dir: tempfile::TempDir,
        progress_tx: &Sender<RemasterStatus>,
        result_tx: &Sender<RemasterOutput>,
        mode: UpscaleMode,
        cancel_flag: &AtomicBool,
        ddim_steps: u32,
        progressive: bool,
        full_parallel: bool,
    ) -> Result<(), String> {
        use std::sync::atomic::AtomicI32;

        ensure_not_cancelled(cancel_flag)?;

        let num_engines = self.engines.len();
        if num_engines == 0 {
            return Err("No upsampling engine found".into());
        }

        let total_jobs = jobs.len();
        if total_jobs == 0 {
            return if cancellation_requested(cancel_flag) {
                Err(cancelled_error())
            } else {
                let _ = progress_tx.send(RemasterStatus::Complete);
                Ok(())
            };
        }

        let primary_counts_by_job: Vec<i32> = jobs
            .iter()
            .map(|job| count_eligible_engines_for_job(job, &self.engines))
            .collect();
        let primary_progress_total: i32 = primary_counts_by_job.iter().sum();
        let progress_counter = AtomicI32::new(0);

        if primary_progress_total == 0 {
            let _ = progress_tx.send(RemasterStatus::Log(
                "Quinlight Audio: skipped AI remastering because no selected engine supports the original sample rates"
                    .into(),
            ));
            let _ = progress_tx.send(RemasterStatus::Complete);
            return Ok(());
        }

        let _ = progress_tx.send(RemasterStatus::Processing {
            current: 0,
            total: primary_progress_total,
            sample_name: format!(
                "Quinlight Audio: {} engines, {} eligible tasks across {} samples",
                num_engines, primary_progress_total, total_jobs
            ),
        });

        let pending_outputs =
            std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
                i32,
                PendingSampleOutputs,
            >::new()));

        let pass1_dir = work_dir.path().join("pass1");

        run_engine_wave(
            &self.engines,
            &jobs,
            None,
            &pass1_dir,
            progress_tx,
            result_tx,
            &progress_counter,
            primary_progress_total,
            &pending_outputs,
            &primary_counts_by_job,
            mode,
            cancel_flag,
            ddim_steps,
            progressive,
            full_parallel,
        )?;

        // Fallback wave: for samples where any primary engine scored below
        // the 0.9 usability floor, run the installed-but-disabled engines
        // to strengthen consensus. Scoped per-sample to minimize extra work.
        if !self.fallback_engines.is_empty() && !cancellation_requested(cancel_flag) {
            let fallback_info = snapshot_failing_samples_and_bump_totals(
                &pending_outputs,
                &jobs,
                &self.fallback_engines,
            );

            if fallback_info.total_additional_tasks > 0 {
                let new_total = primary_progress_total + fallback_info.total_additional_tasks;
                let failing_set: std::collections::HashSet<i32> = fallback_info
                    .failing_sample_indices
                    .iter()
                    .copied()
                    .collect();

                let _ = progress_tx.send(RemasterStatus::Log(format!(
                    "Quinlight Audio: fallback wave — {} sample(s) had at least one primary engine \
                     score below {:.2}; running {} disabled engine(s)",
                    fallback_info.failing_sample_indices.len(),
                    quinlight_usable_score_floor(),
                    self.fallback_engines.len()
                )));
                let _ = progress_tx.send(RemasterStatus::Processing {
                    current: progress_counter.load(std::sync::atomic::Ordering::Relaxed),
                    total: new_total,
                    sample_name: format!(
                        "Quinlight Audio: fallback wave — {} engines, {} extra tasks across \
                         {} sample(s)",
                        self.fallback_engines.len(),
                        fallback_info.total_additional_tasks,
                        fallback_info.failing_sample_indices.len()
                    ),
                });

                let pass2_dir = work_dir.path().join("pass2");
                run_engine_wave(
                    &self.fallback_engines,
                    &jobs,
                    Some(&failing_set),
                    &pass2_dir,
                    progress_tx,
                    result_tx,
                    &progress_counter,
                    new_total,
                    &pending_outputs,
                    &fallback_info.counts_by_job,
                    mode,
                    cancel_flag,
                    ddim_steps,
                    progressive,
                    full_parallel,
                )?;
            }
        }

        ensure_not_cancelled(cancel_flag)?;
        let _ = progress_tx.send(RemasterStatus::Complete);
        Ok(())
    }
}

/// Compute the maximum number of engines that can safely run in parallel
/// given an available-memory figure in MB and a per-model budget in MB.
///
/// When `memory_mb == 0` (detection failed or unsupported platform), returns
/// `num_engines` — the "trust the user" fallback. Otherwise returns
/// `memory_mb / budget_per_model_mb`, clamped to `[1, num_engines]` so a
/// critically low memory figure still admits one engine (running serially
/// rather than erroring out).
fn max_parallel_from_memory(memory_mb: u64, budget_per_model_mb: u64, num_engines: usize) -> usize {
    if memory_mb == 0 || num_engines == 0 || budget_per_model_mb == 0 {
        return num_engines;
    }
    ((memory_mb / budget_per_model_mb) as usize).clamp(1, num_engines)
}

/// Runs one batch wave of upsampling engines. Engines in `engines` run on
/// `jobs` (or the subset allowed by `restrict_to_samples`, identified by
/// `SampleJob::index`). Produced candidates flow through the shared
/// `pending_outputs` map, so Quinlight consensus builds incrementally as
/// candidates arrive (see `record_engine_completion`).
///
/// Concurrency policy:
/// - Default (`full_parallel = false`): engines run **serially**, one at a
///   time. Hybrid still uses 2 workers per engine (1 GPU + 1 CPU), so the
///   worst case is 2 concurrent PyTorch subprocesses; CpuOnly/GpuOnly see
///   exactly 1. Avoids PyTorch hangs observed when many Python subprocesses
///   race for resources.
/// - `full_parallel = true`: all engines run concurrently, capped by RAM
///   (CpuOnly/Hybrid) and/or VRAM (GpuOnly/Hybrid). If the GPU can't hold
///   all engines' models, engines are processed in chunks that fit the
///   budget; each chunk's engines still run in parallel with each other.
#[allow(clippy::too_many_arguments)]
fn run_engine_wave(
    engines: &[Box<dyn engine::UpsampleEngine>],
    jobs: &[SampleJob],
    restrict_to_samples: Option<&std::collections::HashSet<i32>>,
    work_dir: &Path,
    progress_tx: &Sender<RemasterStatus>,
    result_tx: &Sender<RemasterOutput>,
    progress_counter: &std::sync::atomic::AtomicI32,
    progress_total: i32,
    pending_outputs: &PendingOutputMap,
    eligible_engine_counts_by_job: &[i32],
    mode: UpscaleMode,
    cancel_flag: &AtomicBool,
    ddim_steps: u32,
    progressive: bool,
    full_parallel: bool,
) -> Result<(), String> {
    ensure_not_cancelled(cancel_flag)?;

    let num_engines = engines.len();
    if num_engines == 0 {
        return Ok(());
    }

    // Serial by default — parallel engine execution triggers PyTorch
    // subprocess hangs. Guard the probes at the source because
    // `detect_gpu_vram_mb()` spawns `nvidia-smi`; there's no point paying
    // that cost just to divide by a budget we won't consult.
    let max_parallel_engines = if full_parallel {
        const BUDGET_PER_MODEL_MB: u64 = 3072;
        let ram_cap = match mode {
            UpscaleMode::CpuOnly | UpscaleMode::Hybrid => max_parallel_from_memory(
                crate::engine::detect_available_ram_mb(),
                BUDGET_PER_MODEL_MB,
                num_engines,
            ),
            UpscaleMode::GpuOnly => num_engines,
        };
        let vram_cap = match mode {
            UpscaleMode::CpuOnly => num_engines,
            _ => max_parallel_from_memory(
                crate::engine::detect_gpu_vram_mb(),
                BUDGET_PER_MODEL_MB,
                num_engines,
            ),
        };
        ram_cap.min(vram_cap)
    } else {
        1
    };

    // With engines running in parallel, each engine gets a smaller worker
    // pool than in the old serial design. Hybrid splits each engine across
    // one GPU + one CPU worker; GpuOnly/CpuOnly use a single worker per
    // engine.
    let (workers_per_engine, gpu_workers_per_engine) = match mode {
        UpscaleMode::GpuOnly => (1usize, 1usize),
        UpscaleMode::Hybrid => (2usize, 1usize),
        UpscaleMode::CpuOnly => (1usize, 0usize),
    };

    // Deliberately oversubscribe: every concurrent engine subprocess is told
    // to use all logical cores. The kernel scheduler handles the contention.
    // Hypothesis: total wall-clock time drops because whichever engine is
    // currently memory-bound yields to one that's compute-bound, even though
    // per-process efficiency falls due to HT false-sharing and context
    // switches. Paired with staggered nice levels (see run_single_engine) for
    // a CFS weight bias that tends to reduce effective context switches per
    // run.
    let cpu_thread_budget = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);

    // Run engines in chunks sized by the VRAM cap. Each chunk is a separate
    // `thread::scope` so a bad engine in chunk N can short-circuit chunks N+1..
    let engine_errors = std::sync::Mutex::new(Vec::<String>::new());
    let mut chunk_start = 0usize;
    while chunk_start < num_engines {
        ensure_not_cancelled(cancel_flag)?;
        let chunk_end = (chunk_start + max_parallel_engines).min(num_engines);

        std::thread::scope(|outer| {
            for eng_idx in chunk_start..chunk_end {
                let engine = engines[eng_idx].as_ref();
                let errors_ref = &engine_errors;
                outer.spawn(move || {
                    let result = run_single_engine(
                        engine,
                        eng_idx,
                        num_engines,
                        jobs,
                        restrict_to_samples,
                        work_dir,
                        progress_tx,
                        result_tx,
                        progress_counter,
                        progress_total,
                        pending_outputs,
                        eligible_engine_counts_by_job,
                        mode,
                        cancel_flag,
                        ddim_steps,
                        progressive,
                        workers_per_engine,
                        gpu_workers_per_engine,
                        cpu_thread_budget,
                    );
                    if let Err(e) = result {
                        errors_ref.lock().unwrap().push(e);
                    }
                });
            }
        });

        {
            let errs = engine_errors.lock().unwrap();
            if let Some(cancel) = errs.iter().find(|e| is_cancelled_error(e)) {
                return Err(cancel.clone());
            }
            if let Some(first) = errs.first() {
                return Err(first.clone());
            }
        }

        chunk_start = chunk_end;
    }

    ensure_not_cancelled(cancel_flag)?;
    Ok(())
}

/// Runs one engine against `jobs` start-to-finish. Called from inside the
/// outer `thread::scope` in `run_engine_wave` so that multiple engines run
/// concurrently. All shared state (progress counters, pending outputs,
/// channels) is thread-safe; per-engine state (work dirs, work queue,
/// success counters) is local to this call.
#[allow(clippy::too_many_arguments)]
fn run_single_engine(
    engine: &dyn engine::UpsampleEngine,
    eng_idx: usize,
    num_engines: usize,
    jobs: &[SampleJob],
    restrict_to_samples: Option<&std::collections::HashSet<i32>>,
    work_dir: &Path,
    progress_tx: &Sender<RemasterStatus>,
    result_tx: &Sender<RemasterOutput>,
    progress_counter: &std::sync::atomic::AtomicI32,
    progress_total: i32,
    pending_outputs: &PendingOutputMap,
    eligible_engine_counts_by_job: &[i32],
    mode: UpscaleMode,
    cancel_flag: &AtomicBool,
    ddim_steps: u32,
    progressive: bool,
    workers_per_engine: usize,
    gpu_workers_per_engine: usize,
    cpu_thread_budget: usize,
) -> Result<(), String> {
    use std::sync::atomic::{AtomicI32, Ordering};

    let tag = format!("[{}]", engine.name());
    let engine_label = format!("{} ({}/{})", engine.name(), eng_idx + 1, num_engines);
    eprintln!("{tag} starting {engine_label}");

    let success_counter = AtomicI32::new(0);
    let mut eligible_job_indices: Vec<usize> = Vec::new();
    let mut eng_uncached: Vec<usize> = Vec::new();
    let mut eng_cache_hits = 0usize;

    for (job_idx, job) in jobs.iter().enumerate() {
        ensure_not_cancelled(cancel_flag)?;
        if let Some(allowed) = restrict_to_samples
            && !allowed.contains(&job.index)
        {
            continue;
        }
        let original_rate_hz = job.rate.max(0) as u32;
        if !engine.supports_original_rate(original_rate_hz) {
            let skip_message = format!(
                "{} skipped {} ({} Hz >= {} Hz output rate)",
                engine.name(),
                job.display_name(),
                original_rate_hz,
                engine.output_rate()
            );
            eprintln!("{tag} {skip_message}");
            let _ = progress_tx.send(RemasterStatus::Log(skip_message));
            continue;
        }
        eligible_job_indices.push(job_idx);
        let ecache_key = engine_cache_key_for_job(job, engine.cache_id(), ddim_steps as u16);
        if let Some(mut cached) = cache_lookup(
            &job.pcm_sha256,
            engine.cache_id(),
            &ecache_key,
            ddim_steps as u16,
            engine.name(),
            job.index,
            job.output_sample_rate_hz,
        ) {
            eprintln!("{tag} cache hit for {}", job.display_name());
            if job.looped {
                let discovered = search_all_loops(
                    &cached.data,
                    cached.channels.max(1) as usize,
                    job.loop_info,
                    job.rate as u32,
                    48_000,
                );
                log_loop_search_result(
                    &job.display_name(),
                    engine.name(),
                    job.loop_info,
                    job.rate as u32,
                    48_000,
                    &discovered,
                );
                cached.discovered_loops = Some(discovered);
            }
            record_engine_completion(
                job,
                Some(cached),
                progress_counter,
                progress_total,
                progress_tx,
                result_tx,
                pending_outputs,
                format!("{}: {} (cached)", engine_label, job.display_name()),
                eligible_engine_counts_by_job[job_idx],
                cancel_flag,
                progressive,
            );
            success_counter.fetch_add(1, Ordering::Relaxed);
            eng_cache_hits += 1;
        } else {
            eng_uncached.push(job_idx);
        }
    }
    let _ = progress_tx.send(RemasterStatus::Log(format!(
        "{engine_label}: {} to process, {} cached",
        eng_uncached.len(),
        eng_cache_hits
    )));
    ensure_not_cancelled(cancel_flag)?;

    if eligible_job_indices.is_empty() {
        eprintln!("{tag} skipped all {} samples", jobs.len());
        let _ = progress_tx.send(RemasterStatus::Log(format!(
            "{}: 0/0 succeeded",
            engine.name()
        )));
        return Ok(());
    }

    let eng_dir = work_dir.join(format!("engine_{}", engine.name().to_lowercase()));
    let max_batch = engine.max_batch_size();

    // VRAM-aware batch cap for Intel XPU with limited local memory
    let max_batch = if *crate::engine::GPU_VENDOR.get_or_init(crate::engine::detect_gpu)
        == crate::engine::GpuVendor::Intel
        && mode != UpscaleMode::CpuOnly
    {
        let vram = crate::engine::detect_xpu_vram_mb();
        if vram > 0 && vram < 8192 {
            max_batch.min(4)
        } else {
            max_batch
        }
    } else {
        max_batch
    };

    // Don't spin up more workers than we have work to do.
    let actual_workers = workers_per_engine.min(eng_uncached.len()).max(1);
    let gpu_workers = gpu_workers_per_engine.min(actual_workers);

    struct WorkerConfig {
        worker_id: usize,
        base_output_dir: PathBuf,
        device: &'static str,
    }

    let eng_input_dir = eng_dir.join("inputs");
    std::fs::create_dir_all(&eng_input_dir)
        .map_err(|e| format!("Failed to create engine input dir: {e}"))?;
    for &job_idx in &eng_uncached {
        ensure_not_cancelled(cancel_flag)?;
        let job = &jobs[job_idx];
        let conditioning_inputs = job.engine_inputs();
        for input in conditioning_inputs.iter() {
            let filename = input.input_path.file_name().ok_or_else(|| {
                format!("Invalid input path (no filename): {:?}", input.input_path)
            })?;
            let dest = eng_input_dir.join(filename);
            std::fs::copy(&input.input_path, &dest)
                .map_err(|e| format!("Failed to copy WAV to engine input dir: {e}"))?;
        }
    }

    let work_queue = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
        eng_uncached.clone(),
    )));

    let workers: Vec<WorkerConfig> = (0..actual_workers)
        .map(|w| {
            let device = match mode {
                UpscaleMode::GpuOnly => crate::engine::gpu_device_string(),
                UpscaleMode::Hybrid if w < gpu_workers => crate::engine::gpu_device_string(),
                _ => "cpu",
            };
            WorkerConfig {
                worker_id: w,
                base_output_dir: eng_dir.join(format!("worker_{w}_out")),
                device,
            }
        })
        .collect();

    for worker in &workers {
        std::fs::create_dir_all(&worker.base_output_dir)
            .map_err(|e| format!("Failed to create worker output dir: {e}"))?;
    }

    let all_processed = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::<
        usize,
    >::new()));

    // Stagger nice levels starting at the baseline 15: each successive
    // engine steps further into idle territory (capped at 19, the kernel
    // max for unprivileged users). With N=1 this is a no-op; N=4 yields
    // [15, 16, 17, 18]; beyond N=5 engines 4+ pin at 19.
    let engine_nice: i32 = (15i32 + eng_idx as i32).min(19);

    std::thread::scope(|s| {
        for worker in &workers {
            let queue = work_queue.clone();
            let worker_processed = all_processed.clone();
            let eng_input_dir = &eng_input_dir;
            let engine_label = &engine_label;
            let tag = &tag;
            let success_counter = &success_counter;
            s.spawn(move || {
                if cancellation_requested(cancel_flag) {
                    return;
                }
                let mut processed = std::collections::HashSet::new();
                let mut round = 0usize;

                loop {
                    if cancellation_requested(cancel_flag) {
                        break;
                    }

                    let chunk: Vec<usize> = {
                        let mut q = queue.lock().unwrap();
                        let n = max_batch.min(q.len());
                        q.drain(..n).collect()
                    };
                    if chunk.is_empty() {
                        break;
                    }

                    let round_output_dir = worker.base_output_dir.join(format!("round_{round}"));
                    let _ = std::fs::create_dir_all(&round_output_dir);

                    let manifest_path = eng_input_dir
                        .join(format!("manifest_w{}_r{}.json", worker.worker_id, round));
                    let manifest_items: Vec<EngineBatchItem> = chunk
                        .iter()
                        .flat_map(|&job_idx| {
                            let job = &jobs[job_idx];
                            let conditioning_inputs = job.engine_inputs();
                            conditioning_inputs.iter().filter_map(move |input| {
                                let filename = input.input_path.file_name()?;
                                let mut copied_input = input.clone();
                                copied_input.input_path = eng_input_dir.join(filename);
                                let item = engine_batch_item_for_input(job, &copied_input);
                                Some(item)
                            })
                        })
                        .collect();
                    if write_engine_batch_manifest(&manifest_path, manifest_items).is_err() {
                        eprintln!(
                            "{tag} worker {} failed to write round {round} engine manifest",
                            worker.worker_id
                        );
                        let mut q = queue.lock().unwrap();
                        for idx in chunk.into_iter().rev() {
                            q.push_front(idx);
                        }
                        continue;
                    }

                    let run_round = |device: &str,
                                     processed: &mut std::collections::HashSet<usize>|
                     -> Result<(), String> {
                        ensure_not_cancelled(cancel_flag)?;
                        let mut child = engine.spawn_batch(
                            &manifest_path,
                            &round_output_dir,
                            device,
                            ddim_steps,
                            cpu_thread_budget,
                        )?;
                        nice_child(&child, engine_nice);

                        let stderr_thread = child.stderr.take().map(|stderr| {
                            std::thread::spawn(move || {
                                let mut buf = String::new();
                                let mut reader = std::io::BufReader::new(stderr);
                                std::io::Read::read_to_string(&mut reader, &mut buf).ok();
                                buf
                            })
                        });

                        watch_batch_outputs(
                            &mut child,
                            &round_output_dir,
                            &chunk,
                            jobs,
                            engine,
                            ddim_steps,
                            engine_label,
                            progress_counter,
                            progress_total,
                            progress_tx,
                            result_tx,
                            processed,
                            pending_outputs,
                            eligible_engine_counts_by_job,
                            success_counter,
                            cancel_flag,
                            progressive,
                        )?;

                        if cancellation_requested(cancel_flag) {
                            kill_child_best_effort(&mut child);
                            return Err(cancelled_error());
                        }

                        let status = child.wait().map_err(|e| format!("Engine wait: {e}"))?;
                        ensure_not_cancelled(cancel_flag)?;

                        let stderr_tail = stderr_thread
                            .and_then(|t| t.join().ok())
                            .unwrap_or_default();

                        if status.success() {
                            if !stderr_tail.is_empty() {
                                eprintln!(
                                    "{tag} {device} round {round} stderr (exit ok):\n{stderr_tail}"
                                );
                            }
                            Ok(())
                        } else {
                            if !stderr_tail.is_empty() {
                                eprintln!("{tag} {device} round {round} stderr:\n{stderr_tail}");
                            }
                            Err(format!(
                                "{} {device} round {round} exited with {status}",
                                engine.name()
                            ))
                        }
                    };

                    let pre_count = processed.len();
                    let r = run_round(worker.device, &mut processed);
                    if r.as_ref().err().is_some_and(|e| is_cancelled_error(e)) {
                        break;
                    }
                    let none_succeeded = processed.len() == pre_count;
                    if worker.device != "cpu" && (r.is_err() || none_succeeded) {
                        eprintln!(
                            "{tag} GPU worker {} round {round} failed ({}), retrying on CPU",
                            worker.worker_id,
                            if let Err(ref e) = r {
                                e.as_str()
                            } else {
                                "no output produced"
                            }
                        );
                        let _ = std::fs::remove_dir_all(&round_output_dir);
                        let _ = std::fs::create_dir_all(&round_output_dir);
                        if let Err(e) = run_round("cpu", &mut processed)
                            && !is_cancelled_error(&e)
                        {
                            eprintln!(
                                "{tag} CPU retry for worker {} round {round} also failed: {e}",
                                worker.worker_id
                            );
                        }
                    }

                    round += 1;
                }

                {
                    let mut global = worker_processed.lock().unwrap();
                    global.extend(processed.iter().copied());
                }
            });
        }
    });

    if !cancellation_requested(cancel_flag) {
        let processed = all_processed.lock().unwrap();
        let mut engine_failures: Vec<String> = Vec::new();
        for &job_idx in &eng_uncached {
            if !processed.contains(&job_idx) {
                let job = &jobs[job_idx];
                engine_failures.push(format!(
                    "{}: no output for {}",
                    engine.name(),
                    job.display_name()
                ));
            }
        }
        if !engine_failures.is_empty() {
            let msg = format!(
                "{} failed to produce output for {} sample(s): {}",
                engine.name(),
                engine_failures.len(),
                engine_failures.join(", ")
            );
            eprintln!("{tag} FATAL: {msg}");
            return Err(msg);
        }
    }
    ensure_not_cancelled(cancel_flag)?;

    let produced = success_counter.load(Ordering::Relaxed) as usize;
    eprintln!(
        "{tag} produced {}/{} results ({} cached)",
        produced,
        eligible_job_indices.len(),
        eng_cache_hits.min(produced),
    );
    let _ = progress_tx.send(RemasterStatus::Log(format!(
        "{}: {}/{} succeeded",
        engine.name(),
        produced,
        eligible_job_indices.len()
    )));
    Ok(())
}

/// True if `candidate_data` scores below the usability floor against `job`'s
/// native-rate original. Mirrors the score path used by
/// `select_quinlight_mix_internal` — keep them in sync.
fn candidate_fails_gate(job: &SampleJob, candidate_channels: i32, candidate_data: &[f64]) -> bool {
    if job.channels <= 0 || candidate_channels != job.channels || candidate_data.is_empty() {
        return false;
    }
    let score = crate::engine::spectral::spectral_correlation_across_rates(
        &job.original_data,
        job.rate.max(0) as u32,
        candidate_data,
        48_000,
        candidate_channels as usize,
        job.looped,
    );
    score < quinlight_usable_score_floor()
}

/// True if any candidate stored for this sample scored below the usability
/// floor. Used in tests; production code calls `candidate_fails_gate`
/// directly from the lock-released section of the snapshot helper.
#[cfg(test)]
fn sample_has_gate_failure(job: &SampleJob, candidates: &[SampleResult]) -> bool {
    candidates
        .iter()
        .any(|c| candidate_fails_gate(job, c.channels, &c.data))
}

/// Per-sample summary produced by `snapshot_failing_samples_and_bump_totals`,
/// sized/indexed so that callers can feed `counts_by_job` straight back into
/// `run_engine_wave` as its `eligible_engine_counts_by_job` argument.
struct FallbackWaveInfo {
    /// `SampleJob::index` values of samples whose primary candidates include
    /// at least one score below `QUINLIGHT_USABLE_SCORE_FLOOR`.
    failing_sample_indices: Vec<i32>,
    /// Indexed in parallel with `jobs`: number of fallback engines eligible
    /// for each job's rate. Zero for jobs that don't need fallback.
    counts_by_job: Vec<i32>,
    /// Sum of `counts_by_job` — number of engine/sample pairs the fallback
    /// wave will dispatch.
    total_additional_tasks: i32,
}

/// Finds samples where at least one primary engine scored below the usability
/// floor and additively bumps each failing sample's `engines_total` by the
/// count of `fallback_engines` eligible for that sample's rate.
///
/// The candidate data is snapshotted under `pending_outputs`'s lock, then the
/// lock is released for the CPU-heavy spectral scoring, and re-acquired only
/// to apply the `engines_total` bump. Keeping the lock short matters because
/// `record_engine_completion` is the writer that would otherwise block; the
/// atomic bump before the fallback wave dispatches still prevents
/// non-progressive-mode Finals from firing against the unbumped total.
fn snapshot_failing_samples_and_bump_totals(
    pending_outputs: &PendingOutputMap,
    jobs: &[SampleJob],
    fallback_engines: &[Box<dyn engine::UpsampleEngine>],
) -> FallbackWaveInfo {
    let mut info = FallbackWaveInfo {
        failing_sample_indices: Vec::new(),
        counts_by_job: vec![0i32; jobs.len()],
        total_additional_tasks: 0,
    };
    if fallback_engines.is_empty() {
        return info;
    }

    // Step 1: copy candidate (channels, data) tuples out of pending_outputs
    // so we can release the lock before doing FFT work.
    struct CandidateView {
        channels: i32,
        data: Vec<f64>,
    }
    struct SampleSnapshot {
        job_idx: usize,
        candidates: Vec<CandidateView>,
    }
    let snapshots: Vec<SampleSnapshot> = {
        let pending = pending_outputs.lock().unwrap();
        jobs.iter()
            .enumerate()
            .filter_map(|(job_idx, job)| {
                let sample = pending.get(&job.index)?;
                let candidates = sample
                    .candidates
                    .iter()
                    .map(|c| CandidateView {
                        channels: c.channels,
                        data: c.data.clone(),
                    })
                    .collect();
                Some(SampleSnapshot {
                    job_idx,
                    candidates,
                })
            })
            .collect()
    };

    // Step 2: score outside the lock.
    let failing_job_indices: Vec<usize> = snapshots
        .iter()
        .filter(|snap| {
            let job = &jobs[snap.job_idx];
            snap.candidates
                .iter()
                .any(|c| candidate_fails_gate(job, c.channels, &c.data))
        })
        .map(|snap| snap.job_idx)
        .collect();

    if failing_job_indices.is_empty() {
        return info;
    }

    // Step 3: re-acquire lock and bump engines_total for failing samples
    // whose rate is supported by at least one fallback engine. This must
    // happen before the fallback wave dispatches so record_engine_completion
    // doesn't finalize samples against the unbumped total.
    let mut pending = pending_outputs.lock().unwrap();
    for job_idx in failing_job_indices {
        let job = &jobs[job_idx];
        let Some(sample) = pending.get_mut(&job.index) else {
            continue;
        };
        let extra = count_eligible_engines_for_job(job, fallback_engines);
        if extra == 0 {
            continue;
        }
        sample.engines_total += extra;
        info.failing_sample_indices.push(job.index);
        info.counts_by_job[job_idx] = extra;
        info.total_additional_tasks += extra;
    }
    info
}

const QUINLIGHT_NAME: &str = "Quinlight Audio";
const QUINLIGHT_ORIGINAL_NAME: &str = "Original";

/// Default minimum per-engine spectral-correlation score for that engine's
/// output to contribute to Quinlight's consensus. Engines scoring below the
/// floor are dropped for that sample; if fewer than 2 engines remain usable,
/// the original sample is kept. Configurable at runtime via
/// `set_quinlight_usable_score_floor` (e.g. from the `--threshold` CLI flag).
pub const QUINLIGHT_DEFAULT_USABLE_SCORE_FLOOR: f64 = 0.9;

// Runtime-configurable override of the usable-score floor. Stored as the
// bit-representation of an f64 so we can mutate it atomically without needing
// a mutex. Defaults to `QUINLIGHT_DEFAULT_USABLE_SCORE_FLOOR`.
static QUINLIGHT_USABLE_SCORE_FLOOR_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(QUINLIGHT_DEFAULT_USABLE_SCORE_FLOOR.to_bits());

/// Effective usable-score floor for this process. Reads the runtime-set
/// value (via `set_quinlight_usable_score_floor`) or falls back to
/// `QUINLIGHT_DEFAULT_USABLE_SCORE_FLOOR`.
pub fn quinlight_usable_score_floor() -> f64 {
    f64::from_bits(QUINLIGHT_USABLE_SCORE_FLOOR_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Override the usable-score floor for the rest of this process. Called from
/// the CLI layer (e.g. `quinlight upsample --threshold 0.75`). The value is
/// clamped to `[0.0, 1.0]`; callers that pass outside that range get clamped
/// silently.
pub fn set_quinlight_usable_score_floor(value: f64) {
    let clamped = value.clamp(0.0, 1.0);
    QUINLIGHT_USABLE_SCORE_FLOOR_BITS
        .store(clamped.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Returns true if the engine name indicates the original sample was kept —
/// either no AI engines reached consensus, or the loop quality gate rejected
/// the AI output.
pub fn is_no_consensus_result(engine_name: &str) -> bool {
    engine_name == QUINLIGHT_NAME || engine_name.ends_with("(loop gate)")
}

fn engine_short_code(name: &str) -> &str {
    match name {
        "AudioSR" => "A",
        "LavaSR" => "L",
        "FLowHigh" => "F",
        "AP-BWE" => "B",
        "Original" => "O",
        _ => "?",
    }
}

fn quinlight_name_with_contributors(contributors: &[QuinlightContributor]) -> String {
    let codes: Vec<&str> = contributors
        .iter()
        .map(|c| engine_short_code(&c.name))
        .collect();
    if codes.is_empty() {
        QUINLIGHT_NAME.to_string()
    } else {
        format!("{} ({})", QUINLIGHT_NAME, codes.join("+"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuinlightContributor {
    pub name: String,
    pub weight: f64,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuinlightMix {
    pub data: Vec<f64>,
    pub length_frames: i64,
    pub channels: i32,
    pub name: String,
    pub contributors: Vec<QuinlightContributor>,
}

#[derive(Clone)]
struct ScoredEngine<'a> {
    name: &'a str,
    data: &'a [f64],
    #[allow(dead_code)]
    length_frames: i64,
    #[allow(dead_code)]
    channels: i32,
    raw_score: f64,
}

fn scored_engine_cmp(a: &ScoredEngine<'_>, b: &ScoredEngine<'_>) -> std::cmp::Ordering {
    b.raw_score
        .total_cmp(&a.raw_score)
        .then_with(|| {
            crate::engine::engine_preference_rank(a.name)
                .cmp(&crate::engine::engine_preference_rank(b.name))
        })
        .then_with(|| a.name.cmp(b.name))
}

struct QuinlightSelectionReference<'a> {
    target_rms: f64,
    source_native: Option<&'a [f64]>,
    score_48k: Option<&'a [f64]>,
    fallback_48k: Option<&'a [f64]>,
}

fn spectral_intersection_blend(
    usable: &[ScoredEngine<'_>],
    reference_channels: i32,
    looped: bool,
    source_reference: Option<&[f64]>,
    original_rate: u32,
    target_rms: f64,
) -> QuinlightMix {
    let channels = reference_channels.max(1) as usize;
    let min_len = usable.iter().map(|e| e.data.len()).min().unwrap_or(0);
    let min_frames = min_len / channels;

    let names: Vec<&str> = usable.iter().map(|e| e.name).collect();
    eprintln!(
        "  Quinlight: spectral intersection of {} engines: {}",
        usable.len(),
        names
            .iter()
            .zip(usable.iter())
            .map(|(n, e)| format!("{n} (score {:.4})", e.raw_score))
            .collect::<Vec<_>>()
            .join(", "),
    );

    let mut output = vec![0.0f64; min_frames * channels];

    for ch in 0..channels {
        let per_engine: Vec<Vec<f64>> = usable
            .iter()
            .map(|e| {
                e.data
                    .iter()
                    .skip(ch)
                    .step_by(channels)
                    .take(min_frames)
                    .copied()
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = per_engine.iter().map(|v| v.as_slice()).collect();
        let intersected = crate::engine::spectral_intersection(&refs, looped);
        for (i, &sample) in intersected.iter().take(min_frames).enumerate() {
            output[i * channels + ch] = sample;
        }
    }

    if let Some(source_reference) = source_reference {
        output = apply_source_frequency_blend_interleaved(
            &output,
            source_reference,
            original_rate,
            48_000,
            channels,
            looped,
        );
    }

    let out_rms = rms_or_zero(&output);
    if out_rms > 1e-8 && target_rms > 1e-8 {
        let gain = target_rms / out_rms;
        for sample in &mut output {
            *sample *= gain;
        }
    }

    let weight = 1.0 / usable.len() as f64;
    let contributors: Vec<QuinlightContributor> = usable
        .iter()
        .map(|e| QuinlightContributor {
            name: e.name.to_string(),
            weight,
            score: e.raw_score,
        })
        .collect();
    let name = quinlight_name_with_contributors(&contributors);
    QuinlightMix {
        data: output,
        length_frames: min_frames as i64,
        channels: reference_channels,
        name,
        contributors,
    }
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
struct PreparedSampleData {
    reference_data: Vec<f64>,
    hold_reference_data: Option<Vec<f64>>,
    // For sustain-packed samples, this is the fully sustain-aligned post-keyoff
    // one-copy sample used to build the release-side model input.
    release_reference_data: Option<Vec<f64>>,
    #[allow(dead_code)] // Read only from tests
    model_input: Vec<f64>,
    release_model_input: Option<Vec<f64>>,
    conditioning_input_48k: Vec<f64>,
    conditioning_input_48k_frames: i64,
    /// For sustain-packed samples, the release segment resampled to 48 kHz
    /// independently from the hold segment to avoid concatenation artifacts.
    conditioning_input_48k_release: Option<Vec<f64>>,
    conditioning_input_48k_release_frames: i64,
    #[allow(dead_code)] // Read only from tests
    input_length_frames: i64,
    input_layout: PreparedInputLayout,
    target_rms: f64,
}

const CONDITIONING_TARGET_PEAK: f64 = 0.5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChannelInput {
    pub channel_index: usize,
    pub channel_name: String,
    pub input_path: PathBuf,
    pub input_length_frames: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct EngineBatchManifest {
    version: u8,
    items: Vec<EngineBatchItem>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct EngineBatchItem {
    stem: String,
    sample_index: i32,
    source_stem: String,
    conditioning_wav_path: String,
    original_rate_hz: u32,
    original_nyquist_hz: f64,
    conditioning_rate_hz: u32,
    conditioning_lowpass_hz: f64,
    source_channels: i32,
    conditioning_channels: i32,
    channel_index: usize,
    channel_name: String,
    looped: bool,
    source_length_frames: i64,
    effective_length_frames: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopPrepMode {
    OneShot,
    BoundaryLoop,
    SustainPacked,
    WholeSampleLoopFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoopPrepPlan {
    mode: LoopPrepMode,
    source_length_frames: i64,
    saved_length_frames: i64,
    normal_loop: SampleLoopRegion,
    sustain_loop: SampleLoopRegion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedInputLayout {
    Single {
        input_frames: i64,
    },
    Sustain {
        hold_input_frames: i64,
        release_input_frames: i64,
    },
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
#[derive(Debug, Clone)]
struct LoopPreparedReference {
    final_reference: Vec<f64>,
    hold_reference: Option<Vec<f64>>,
    // For sustain-packed samples, this is the fully sustain-aligned post-keyoff
    // sample used to build the release-side model input.
    release_reference: Option<Vec<f64>>,
}

fn original_nyquist_hz(original_rate_hz: u32) -> f64 {
    original_rate_hz as f64 * 0.5
}

fn conditioning_lowpass_hz(original_rate_hz: u32) -> f64 {
    original_nyquist_hz(original_rate_hz) * 0.90
}

fn apply_source_frequency_blend_interleaved(
    candidate: &[f64],
    source_reference: &[f64],
    source_rate_hz: u32,
    output_rate_hz: u32,
    channels: usize,
    looped: bool,
) -> Vec<f64> {
    if channels == 0 || candidate.is_empty() || source_reference.is_empty() {
        return candidate.to_vec();
    }
    if source_rate_hz < 2 || output_rate_hz < 2 {
        return candidate.to_vec();
    }

    // Cross-rate STFT bin phases don't line up (different FFT sizes assign
    // different phases to the same frequency), so resample the source to the
    // candidate's rate first. We use the one-shot resampler even for looped
    // samples: `resample_audio` with `LoopAware` tiles the *entire* signal
    // (treating the whole sample as loop body), which disagrees with the
    // caller's `resample_segment_reference` that tiles only the loop region,
    // and empirically adds seam noise on real fixtures.
    let source_at_output_rate = if source_rate_hz == output_rate_hz {
        source_reference.to_vec()
    } else {
        match resample_audio_one_shot(source_reference, source_rate_hz, output_rate_hz, channels) {
            Ok(data) if !data.is_empty() => data,
            _ => return candidate.to_vec(),
        }
    };

    if channels == 1 {
        return crate::engine::spectral::apply_source_frequency_blend(
            candidate,
            &source_at_output_rate,
            source_rate_hz,
            output_rate_hz,
            looped,
        );
    }

    if channels == 2 {
        let (cand_l, cand_r) = simd::deinterleave_stereo_f64(candidate);
        let (src_l, src_r) = simd::deinterleave_stereo_f64(&source_at_output_rate);
        let blended_l = crate::engine::spectral::apply_source_frequency_blend(
            &cand_l,
            &src_l,
            source_rate_hz,
            output_rate_hz,
            looped,
        );
        let blended_r = crate::engine::spectral::apply_source_frequency_blend(
            &cand_r,
            &src_r,
            source_rate_hz,
            output_rate_hz,
            looped,
        );
        return simd::interleave_stereo_f64(&blended_l, &blended_r);
    }

    let blended_channels: Vec<Vec<f64>> = (0..channels)
        .map(|ch| {
            let cand_ch: Vec<f64> = candidate
                .iter()
                .skip(ch)
                .step_by(channels)
                .copied()
                .collect();
            let src_ch: Vec<f64> = source_at_output_rate
                .iter()
                .skip(ch)
                .step_by(channels)
                .copied()
                .collect();
            crate::engine::spectral::apply_source_frequency_blend(
                &cand_ch,
                &src_ch,
                source_rate_hz,
                output_rate_hz,
                looped,
            )
        })
        .collect();

    let frames = blended_channels
        .iter()
        .map(|channel| channel.len())
        .min()
        .unwrap_or(0);
    let mut blended = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for channel in &blended_channels {
            blended.push(channel[frame]);
        }
    }
    blended
}

fn clamp_loop_region(region: SampleLoopRegion, source_length_frames: i64) -> SampleLoopRegion {
    if region.mode == SampleLoopMode::None || source_length_frames <= 0 {
        return SampleLoopRegion::none();
    }

    let start_frames = region.start_frames.clamp(0, source_length_frames);
    let end_frames = region.end_frames.clamp(0, source_length_frames);
    if end_frames <= start_frames {
        SampleLoopRegion::none()
    } else {
        SampleLoopRegion {
            start_frames,
            end_frames,
            mode: region.mode,
        }
    }
}

fn ping_pong_loop_is_degenerate(region: SampleLoopRegion) -> bool {
    region.mode == SampleLoopMode::PingPong && (region.end_frames - region.start_frames) < 2
}

fn compute_saved_length_frames(
    source_length_frames: i64,
    normal_loop: SampleLoopRegion,
    sustain_loop: SampleLoopRegion,
) -> i64 {
    match (normal_loop.has_loop(), sustain_loop.has_loop()) {
        (false, false) => source_length_frames,
        (true, false) => normal_loop.end_frames,
        (false, true) => source_length_frames,
        (true, true) => normal_loop.end_frames.max(sustain_loop.end_frames),
    }
}

impl LoopPrepPlan {
    fn from_sample(source_length_frames: i64, loop_info: SampleLoopInfo) -> Self {
        let source_length_frames = source_length_frames.max(0);
        let normal_loop = clamp_loop_region(loop_info.normal, source_length_frames);
        let sustain_loop = clamp_loop_region(loop_info.sustain, source_length_frames);
        let raw_has_invalid_loop = (loop_info.normal.mode != SampleLoopMode::None
            && !normal_loop.has_loop())
            || (loop_info.sustain.mode != SampleLoopMode::None && !sustain_loop.has_loop());
        let has_degenerate_ping_pong =
            ping_pong_loop_is_degenerate(normal_loop) || ping_pong_loop_is_degenerate(sustain_loop);

        if raw_has_invalid_loop || has_degenerate_ping_pong {
            return Self {
                mode: LoopPrepMode::WholeSampleLoopFallback,
                source_length_frames,
                saved_length_frames: source_length_frames,
                normal_loop,
                sustain_loop,
            };
        }

        if !normal_loop.has_loop() && !sustain_loop.has_loop() {
            return Self {
                mode: LoopPrepMode::OneShot,
                source_length_frames,
                saved_length_frames: source_length_frames,
                normal_loop,
                sustain_loop,
            };
        }

        let saved_length_frames =
            compute_saved_length_frames(source_length_frames, normal_loop, sustain_loop);
        Self {
            mode: if sustain_loop.has_loop() {
                LoopPrepMode::SustainPacked
            } else {
                LoopPrepMode::BoundaryLoop
            },
            source_length_frames,
            saved_length_frames,
            normal_loop,
            sustain_loop,
        }
    }

    fn from_loop_flag(source_length_frames: i64, looped: bool) -> Self {
        if looped {
            Self::from_sample(
                source_length_frames,
                SampleLoopInfo::forward(0, source_length_frames.max(0)),
            )
        } else {
            Self::from_sample(source_length_frames, SampleLoopInfo::none())
        }
    }

    fn is_looped(self) -> bool {
        self.normal_loop.has_loop() || self.sustain_loop.has_loop()
    }

    #[allow(dead_code)]
    fn uses_boundary_loop(self) -> bool {
        matches!(self.mode, LoopPrepMode::BoundaryLoop)
    }

    #[allow(dead_code)]
    fn uses_sustain(self) -> bool {
        matches!(self.mode, LoopPrepMode::SustainPacked)
    }

    #[allow(dead_code)]
    fn uses_fallback(self) -> bool {
        matches!(self.mode, LoopPrepMode::WholeSampleLoopFallback)
    }

    fn pre_keyoff_loop(self) -> SampleLoopRegion {
        if self.sustain_loop.has_loop() {
            self.sustain_loop
        } else {
            self.normal_loop
        }
    }

    fn post_keyoff_loop(self) -> SampleLoopRegion {
        if self.normal_loop.has_loop() {
            self.normal_loop
        } else {
            SampleLoopRegion::none()
        }
    }

    fn primary_loop_start_frames(self) -> i64 {
        let loop_region = self.pre_keyoff_loop();
        if loop_region.has_loop() {
            loop_region.start_frames
        } else {
            0
        }
    }
}

pub(crate) fn sample_frame_count(data: &[f64], channels: usize) -> i64 {
    if channels == 0 {
        0
    } else {
        (data.len() / channels) as i64
    }
}

fn sample_prefix<'a>(data: &'a [f64], channels: usize, frames: i64) -> &'a [f64] {
    if channels == 0 || data.is_empty() {
        return &[];
    }

    let requested_samples = (frames.max(0) as usize).saturating_mul(channels);
    &data[..data.len().min(requested_samples)]
}

fn sample_region<'a>(
    data: &'a [f64],
    channels: usize,
    start_frames: i64,
    end_frames: i64,
) -> &'a [f64] {
    if channels == 0 || data.is_empty() {
        return &[];
    }

    let start_samples = (start_frames.max(0) as usize)
        .saturating_mul(channels)
        .min(data.len());
    let end_samples = (end_frames.max(start_frames) as usize)
        .saturating_mul(channels)
        .min(data.len());
    &data[start_samples..end_samples]
}

fn push_interleaved_frame(data: &[f64], channels: usize, frame_index: usize, dst: &mut Vec<f64>) {
    let start = frame_index.saturating_mul(channels);
    let end = start.saturating_add(channels).min(data.len());
    if start < end {
        dst.extend_from_slice(&data[start..end]);
    }
}

fn build_ping_pong_extension_chunk(
    data: &[f64],
    channels: usize,
    loop_region: SampleLoopRegion,
) -> Vec<f64> {
    let loop_segment = sample_region(
        data,
        channels,
        loop_region.start_frames,
        loop_region.end_frames,
    );
    if channels == 0 || loop_segment.is_empty() {
        return Vec::new();
    }

    let frames = loop_segment.len() / channels;
    if frames < 2 {
        return Vec::new();
    }

    let mut extension = Vec::with_capacity(loop_segment.len().saturating_mul(2));
    for frame in (0..frames - 1).rev() {
        push_interleaved_frame(loop_segment, channels, frame, &mut extension);
    }
    for frame in 1..frames {
        push_interleaved_frame(loop_segment, channels, frame, &mut extension);
    }
    extension
}

fn loop_extension_after_saved(
    data: &[f64],
    channels: usize,
    loop_region: SampleLoopRegion,
) -> Vec<f64> {
    if !loop_region.has_loop() || channels == 0 || data.is_empty() {
        return Vec::new();
    }

    match loop_region.mode {
        SampleLoopMode::Forward => sample_region(
            data,
            channels,
            loop_region.start_frames,
            loop_region.end_frames,
        )
        .to_vec(),
        SampleLoopMode::PingPong => build_ping_pong_extension_chunk(data, channels, loop_region),
        SampleLoopMode::None => Vec::new(),
    }
}

fn minimum_loop_input_samples(
    data: &[f64],
    loop_extension: &[f64],
    requested_min_samples: usize,
) -> usize {
    requested_min_samples.max(
        data.len()
            .saturating_add(loop_extension.len().saturating_mul(2)),
    )
}

#[cfg(test)]
/// Smooth the junctions in a tiled loop layout before SINC resampling.
/// The tiled layout is: [reference(0..loop_end)][loop_body][loop_body]...
/// Each junction has a gap where loop_end-1 ≠ loop_start. A short crossfade
/// at each junction prevents the SINC kernel from ringing at the discontinuity.
/// Only modifies the disposable tiled copy — the original reference is untouched.
fn smooth_tiled_loop_junctions(
    layout: &mut [f64],
    channels: usize,
    first_copy_frames: usize,
    loop_region: SampleLoopRegion,
) {
    if channels == 0 || !loop_region.has_loop() {
        return;
    }
    let loop_start = loop_region.start_frames.max(0) as usize;
    let loop_end = loop_region.end_frames.max(0) as usize;
    if loop_end <= loop_start + 1 {
        return;
    }
    let loop_body_frames = loop_end - loop_start;
    let total_frames = layout.len() / channels;
    // Pitch-aware crossfade window: half a pitch period when detectable (enough
    // to suppress SINC ringing without over-smoothing), else 16 taps.
    let pitch_period = estimate_pitch_period_frames(layout, channels, loop_start, loop_end);
    let window = match pitch_period {
        Some(period) => ((period / 2.0).round() as usize).clamp(8, 512),
        None => 16usize,
    }
    .min(loop_body_frames / 2)
    .max(1);

    // Walk each junction in the tiled layout and smooth the approach.
    // Only modify the "before" side (tail approaching the junction), blending
    // toward the first "after" value (the junction point). Equal-power (sine)
    // fade preserves energy across the junction.
    let mut junction = first_copy_frames;
    while junction < total_frames && junction >= window {
        let junction_idx = junction * channels;
        if junction_idx + channels > layout.len() {
            break;
        }
        let target: Vec<f64> = (0..channels).map(|ch| layout[junction_idx + ch]).collect();
        for i in 0..window {
            let t = (i + 1) as f64 / (window + 1) as f64;
            let w_before = (std::f64::consts::FRAC_PI_2 * (1.0 - t)).sin();
            let w_target = (std::f64::consts::FRAC_PI_2 * t).sin();
            let before_idx = (junction - window + i) * channels;
            for ch in 0..channels {
                layout[before_idx + ch] =
                    layout[before_idx + ch] * w_before + target[ch] * w_target;
            }
        }
        junction += loop_body_frames;
    }
}

fn build_boundary_loop_input(
    data: &[f64],
    channels: usize,
    loop_region: SampleLoopRegion,
    requested_min_samples: usize,
) -> Vec<f64> {
    if channels == 0 || data.is_empty() {
        return Vec::new();
    }

    let loop_extension = loop_extension_after_saved(data, channels, loop_region);
    if loop_extension.is_empty() {
        return data.to_vec();
    }

    let target_len = minimum_loop_input_samples(data, &loop_extension, requested_min_samples);
    let mut repeated = Vec::with_capacity(target_len.max(data.len()));
    repeated.extend_from_slice(data);
    while repeated.len() < target_len {
        let remaining = target_len - repeated.len();
        let copy_len = remaining.min(loop_extension.len());
        repeated.extend_from_slice(&loop_extension[..copy_len]);
    }
    repeated
}

fn loop_body_repair_window_frames(loop_region: SampleLoopRegion) -> i64 {
    let loop_len = (loop_region.end_frames - loop_region.start_frames).max(0);
    match loop_region.mode {
        SampleLoopMode::Forward => (loop_len / 2).min((loop_len - 1).max(0)),
        SampleLoopMode::PingPong => loop_len / 2,
        SampleLoopMode::None => 0,
    }
}

fn apply_delta_ramp_in_place(
    data: &mut [f64],
    channels: usize,
    start_frame: i64,
    end_frame: i64,
    delta: &[f64],
    strongest_at_start: bool,
) {
    if channels == 0 || delta.len() < channels {
        return;
    }

    let frames = sample_frame_count(data, channels);
    let start_frame = start_frame.clamp(0, frames);
    let end_frame = end_frame.clamp(start_frame, frames);
    let window_frames = end_frame - start_frame;
    if window_frames <= 0 {
        return;
    }

    let window_frames_f32 = window_frames as f64;
    for offset in 0..window_frames as usize {
        let frame = start_frame as usize + offset;
        let weight = if strongest_at_start {
            (window_frames as usize - offset) as f64 / window_frames_f32
        } else {
            (offset + 1) as f64 / window_frames_f32
        };
        let sample_index = frame * channels;
        for ch in 0..channels {
            data[sample_index + ch] += delta[ch] * weight;
        }
    }
}

fn repair_forward_loop_tail_in_place(
    data: &mut [f64],
    channels: usize,
    loop_start_frames: i64,
    loop_end_frames: i64,
) {
    if channels == 0 || data.is_empty() {
        return;
    }

    let frames = sample_frame_count(data, channels);
    let loop_start = loop_start_frames.clamp(0, frames);
    let loop_end = loop_end_frames.clamp(loop_start, frames);
    let loop_region = SampleLoopRegion::forward(loop_start, loop_end);
    let window_frames = loop_body_repair_window_frames(loop_region);
    if window_frames <= 0 {
        return;
    }

    let anchor_index = loop_start as usize * channels;
    let boundary_index = (loop_end - 1) as usize * channels;
    let mut delta = vec![0.0f64; channels];
    for ch in 0..channels {
        delta[ch] = data[anchor_index + ch] - data[boundary_index + ch];
    }

    apply_delta_ramp_in_place(
        data,
        channels,
        loop_end - window_frames,
        loop_end,
        &delta,
        false,
    );
}

fn repair_attack_tail_to_loop_head_in_place(
    data: &mut [f64],
    channels: usize,
    loop_start_frames: i64,
) {
    if channels == 0 || data.is_empty() || loop_start_frames <= 0 {
        return;
    }

    let frames = sample_frame_count(data, channels);
    let loop_start = loop_start_frames.clamp(0, frames);
    let window_frames = loop_start / 2;
    if window_frames <= 0 {
        return;
    }

    let anchor_index = loop_start as usize * channels;
    let boundary_index = (loop_start - 1) as usize * channels;
    let mut delta = vec![0.0f64; channels];
    for ch in 0..channels {
        delta[ch] = data[anchor_index + ch] - data[boundary_index + ch];
    }

    apply_delta_ramp_in_place(
        data,
        channels,
        loop_start - window_frames,
        loop_start,
        &delta,
        false,
    );
}

fn repair_release_head_to_loop_tail_in_place(
    release_data: &mut [f64],
    loop_tail_anchor: &[f64],
    channels: usize,
    sustain_end_frames: i64,
) {
    if channels == 0
        || release_data.is_empty()
        || loop_tail_anchor.len() < channels
        || sustain_end_frames < 0
    {
        return;
    }

    let frames = sample_frame_count(release_data, channels);
    let sustain_end = sustain_end_frames.clamp(0, frames);
    let window_frames = (frames - sustain_end).max(0) / 2;
    if window_frames <= 0 || sustain_end >= frames {
        return;
    }

    let boundary_index = sustain_end as usize * channels;
    let mut delta = vec![0.0f64; channels];
    for ch in 0..channels {
        delta[ch] = loop_tail_anchor[ch] - release_data[boundary_index + ch];
    }

    apply_delta_ramp_in_place(
        release_data,
        channels,
        sustain_end,
        sustain_end + window_frames,
        &delta,
        true,
    );
}

fn repair_ping_pong_turnarounds_in_place(
    data: &mut [f64],
    channels: usize,
    loop_start_frames: i64,
    loop_end_frames: i64,
) {
    if channels == 0 || data.is_empty() {
        return;
    }

    let frames = sample_frame_count(data, channels);
    let loop_start = loop_start_frames.clamp(0, frames);
    let loop_end = loop_end_frames.clamp(loop_start, frames);
    let loop_region = SampleLoopRegion::ping_pong(loop_start, loop_end);
    let window_frames = loop_body_repair_window_frames(loop_region);
    if window_frames <= 0 || (loop_end - loop_start) < 2 {
        return;
    }

    let start_index = loop_start as usize * channels;
    let start_anchor_index = (loop_start + 1) as usize * channels;
    let end_index = (loop_end - 1) as usize * channels;
    let end_anchor_index = (loop_end - 2) as usize * channels;
    if end_anchor_index + channels > data.len() {
        return;
    }

    let mut start_delta = vec![0.0f64; channels];
    let mut end_delta = vec![0.0f64; channels];
    for ch in 0..channels {
        start_delta[ch] = data[start_anchor_index + ch] - data[start_index + ch];
        end_delta[ch] = data[end_anchor_index + ch] - data[end_index + ch];
    }

    apply_delta_ramp_in_place(
        data,
        channels,
        loop_end - window_frames,
        loop_end,
        &end_delta,
        false,
    );
    apply_delta_ramp_in_place(
        data,
        channels,
        loop_start,
        loop_start + window_frames,
        &start_delta,
        true,
    );
}

fn repair_loop_body_in_place(data: &mut [f64], channels: usize, loop_region: SampleLoopRegion) {
    match loop_region.mode {
        SampleLoopMode::Forward => repair_forward_loop_tail_in_place(
            data,
            channels,
            loop_region.start_frames,
            loop_region.end_frames,
        ),
        SampleLoopMode::PingPong => repair_ping_pong_turnarounds_in_place(
            data,
            channels,
            loop_region.start_frames,
            loop_region.end_frames,
        ),
        SampleLoopMode::None => {}
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct LoopSeamFixOptions {
    enabled: bool,
    search_radius_frames: i64,
    amplitude_weight: f64,
    slope_weight: f64,
    fallback_score_threshold: f64,
    micro_crossfade_min_frames: usize,
    micro_crossfade_max_frames: usize,
}

#[cfg(test)]
const DEFAULT_LOOP_SEAM_FIX_OPTIONS: LoopSeamFixOptions = LoopSeamFixOptions {
    enabled: true,
    search_radius_frames: 256,
    amplitude_weight: 4.0,
    slope_weight: 1.0,
    fallback_score_threshold: 0.15,
    micro_crossfade_min_frames: 8,
    micro_crossfade_max_frames: 32,
};

#[cfg(test)]
const LOOP_REPAIR_GUARD_WINDOW_FRAMES: usize = 12;
#[cfg(test)]
const LOOP_REPAIR_LOCAL_JUMP_REL_TOLERANCE: f64 = 1.08;
#[cfg(test)]
const LOOP_REPAIR_LOCAL_JUMP_ABS_TOLERANCE: f64 = 0.01;
#[cfg(test)]
const LOOP_REPAIR_GOOD_ENOUGH_SEAM_SCORE: f64 = 0.55;

/// How much worse the AI loop boundary can be compared to the original sample's
/// boundary before rejection. Measured pre-crossfade. The downstream crossfade
/// smooths modest wrap discontinuities, so the gate only needs to catch AI
/// results the crossfade can't rescue — hence generous headroom.
const AI_LOOP_SEAM_GATE_TOLERANCE: f64 = 2.0;

/// Absolute floor for the amplitude gap gate. Allows near-perfect originals
/// (gap ~ 0) to tolerate small AI imperfections without immediate rejection.
const AI_LOOP_SEAM_GATE_FLOOR: f64 = 0.05;

/// Absolute floor for the slope gate (in original-rate-normalized units).
const AI_LOOP_SEAM_SLOPE_FLOOR: f64 = 0.10;

/// How much larger (per window) the AI loop body's peak |slope| may be
/// compared to the rate-normalized original before we treat it as a phase
/// anomaly. 10.0 tolerates HF brightening; peaks more than 10× the original's
/// per-window envelope are treated as inserted content.
const AI_LOOP_BODY_SLOPE_TOLERANCE: f64 = 10.0;

/// Absolute floor for the loop-body slope envelope check; near-flat windows
/// shouldn't reject on quantization-level noise.
const AI_LOOP_BODY_SLOPE_FLOOR: f64 = 0.04;

/// Target window size (at output rate, i.e. 48 kHz) for the loop-body slope
/// envelope. The actual window count scales with the loop body length so
/// short loops still produce multiple windows.
const AI_LOOP_BODY_WINDOW_FRAMES: usize = 32;

/// Max ratio between the peak HF energy near the loop boundary and the
/// loop-body middle's average HF energy in the AI output. Above this
/// factor, the AI has concentrated its added high-frequency content at
/// the loop boundary — typically a tick/chirp/burst captured into the
/// loop. Real ticks sit orders of magnitude above normal loop-body
/// variation; set high enough to distinguish real problems from natural
/// AI variation (e.g. twilight.umx sample #12's tick lands at ~11885×).
const AI_LOOP_HF_CONCENTRATION_TOLERANCE: f64 = 100.0;

/// Floor on the sample-wide average HF energy below which the concentration
/// check is skipped. Prevents absurd ratios on samples the AI intentionally
/// left essentially baseband (nothing above original's Nyquist to add).
const AI_LOOP_HF_CONCENTRATION_FLOOR: f64 = 1.0e-6;

/// Radius (in STFT frames) around loop_start and loop_end used to sample the
/// local HF peak. At the default STFT config (~512-sample hop at 48 kHz),
/// 3 frames ≈ 32 ms on either side of the boundary.
const AI_LOOP_HF_RADIUS_STFT_FRAMES: usize = 3;

/// Minimum STFT-frame count in the AI sample for the HF-concentration gate
/// to engage. Short samples don't have enough spectral resolution to
/// distinguish boundary anomalies from the loop body's own content.
const AI_LOOP_HF_MIN_BODY_STFT_FRAMES: usize = 12;

/// Minimum STFT-frame count in the loop body's "middle" region (i.e., the
/// body minus the boundary radii on both sides) for the concentration
/// comparison to be meaningful. Prevents comparing a boundary max against
/// a one-or-two-frame baseline.
const AI_LOOP_HF_MIN_MIDDLE_STFT_FRAMES: usize = 4;

/// Max adjacent-frame jump allowed anywhere inside the AI loop body
/// *after* the seam crossfade runs. The crossfade's endpoint-force can
/// leave a cliff one sample inside the loop when the AI pre-crossfade
/// tail is far from the loop start. Measured directly from the
/// crossfaded data; has no pre-crossfade equivalent.
const AI_LOOP_POST_CROSSFADE_INNER_JUMP_LIMIT: f64 = 0.15;

/// Max absolute amplitude jump at the forward loop wrap point (end-1 → start).
fn loop_wrap_gap(data: &[f64], channels: usize, loop_start: usize, loop_end: usize) -> f64 {
    if channels == 0 || loop_end <= loop_start + 1 {
        return 0.0;
    }
    let end_idx = (loop_end - 1) * channels;
    let start_idx = loop_start * channels;
    (0..channels)
        .map(|ch| (data[end_idx + ch] - data[start_idx + ch]).abs())
        .fold(0.0f64, f64::max)
}

/// Slope (first derivative) mismatch at the forward loop wrap point.
/// Compares the slope arriving at loop_end-1 with the slope departing loop_start.
/// A large value means the waveform is heading in different directions on each
/// side of the wrap — a phase mismatch even if the amplitudes happen to align.
fn loop_wrap_slope_gap(data: &[f64], channels: usize, loop_start: usize, loop_end: usize) -> f64 {
    let frames = data.len() / channels.max(1);
    if channels == 0 || loop_end <= loop_start + 2 || loop_end < 2 || loop_start + 1 >= frames {
        return 0.0;
    }
    let mut max_gap = 0.0f64;
    for ch in 0..channels {
        let slope_end = data[(loop_end - 1) * channels + ch] - data[(loop_end - 2) * channels + ch];
        let slope_start = data[(loop_start + 1) * channels + ch] - data[loop_start * channels + ch];
        max_gap = max_gap.max((slope_start - slope_end).abs());
    }
    max_gap
}

/// Per-window peak |slope| across the loop body. Slopes are first differences
/// between adjacent frames; peaks are maxed across channels and across frames
/// inside each window. Returns `num_windows` values covering `[loop_start,
/// loop_end)`. Used by the loop-body gate to detect ticks/chirps and phase
/// drift inside the loop that the wrap-point check can't see.
fn loop_body_slope_envelope(
    data: &[f64],
    channels: usize,
    loop_start: usize,
    loop_end: usize,
    num_windows: usize,
) -> Vec<f64> {
    if channels == 0 || loop_end <= loop_start + 1 || num_windows == 0 {
        return Vec::new();
    }
    let total_frames = data.len() / channels;
    let le = loop_end.min(total_frames);
    if le <= loop_start + 1 {
        return Vec::new();
    }
    let body_frames = le - loop_start;
    // `body_frames - 1` adjacent pairs produce slopes.
    let pairs = body_frames - 1;
    if pairs == 0 {
        return vec![0.0; num_windows];
    }

    let mut envelope = Vec::with_capacity(num_windows);
    for w in 0..num_windows {
        // Split `pairs` evenly across `num_windows`; the last window absorbs
        // any remainder from integer division.
        let start_pair = (w * pairs) / num_windows;
        let end_pair = if w + 1 == num_windows {
            pairs
        } else {
            ((w + 1) * pairs) / num_windows
        };
        let mut peak = 0.0f64;
        for p in start_pair..end_pair {
            let frame = loop_start + p;
            for ch in 0..channels {
                let slope = data[(frame + 1) * channels + ch] - data[frame * channels + ch];
                let abs_slope = slope.abs();
                if abs_slope > peak {
                    peak = abs_slope;
                }
            }
        }
        envelope.push(peak);
    }
    envelope
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct LoopBoundaryMetrics {
    seam_score: f64,
    wrap_gap: f64,
    local_adjacent_jump: f64,
}

#[cfg(test)]
fn adjacent_jump_at_runtime(samples: &[f64], channels: usize, frame_index: usize) -> f64 {
    if channels == 0 || frame_index == 0 {
        return 0.0;
    }
    let sample_index = frame_index.saturating_mul(channels);
    if sample_index >= samples.len() {
        return 0.0;
    }
    let prev_index = sample_index.saturating_sub(channels);
    let mut max_gap = 0.0f64;
    for ch in 0..channels {
        max_gap = max_gap.max((samples[sample_index + ch] - samples[prev_index + ch]).abs());
    }
    max_gap
}

#[cfg(test)]
fn max_adjacent_jump_near_loop(
    samples: &[f64],
    channels: usize,
    loop_start_frame: usize,
    loop_end_frame: usize,
    window_frames: usize,
) -> f64 {
    if channels == 0 {
        return 0.0;
    }
    let total_frames = sample_frame_count(samples, channels) as usize;
    if total_frames < 2 {
        return 0.0;
    }

    let mut max_jump = 0.0f64;
    let centers = [loop_start_frame, loop_end_frame.saturating_sub(1)];
    for center in centers {
        let start = center.saturating_sub(window_frames).max(1);
        let end = (center + window_frames).min(total_frames.saturating_sub(1));
        for frame in start..=end {
            max_jump = max_jump.max(adjacent_jump_at_runtime(samples, channels, frame));
        }
    }
    max_jump
}

#[cfg(test)]
fn compute_loop_boundary_metrics(
    samples: &[f64],
    channels: usize,
    loop_start_frame: usize,
    loop_end_frame: usize,
) -> LoopBoundaryMetrics {
    LoopBoundaryMetrics {
        seam_score: seam_score_at_forward_wrap(
            samples,
            channels,
            loop_start_frame,
            loop_end_frame,
            DEFAULT_LOOP_SEAM_FIX_OPTIONS.amplitude_weight,
            DEFAULT_LOOP_SEAM_FIX_OPTIONS.slope_weight,
        ),
        wrap_gap: max_wrap_amplitude_gap(samples, channels, loop_start_frame, loop_end_frame),
        local_adjacent_jump: max_adjacent_jump_near_loop(
            samples,
            channels,
            loop_start_frame,
            loop_end_frame,
            LOOP_REPAIR_GUARD_WINDOW_FRAMES,
        ),
    }
}

#[cfg(test)]
fn apply_boundary_repair_with_guard<F>(
    data: &mut [f64],
    channels: usize,
    loop_region: SampleLoopRegion,
    context: &str,
    mut repair: F,
) -> bool
where
    F: FnMut(&mut [f64]),
{
    if channels == 0 || data.is_empty() || !loop_region.has_loop() {
        return true;
    }

    let total_frames = sample_frame_count(data, channels) as usize;
    if total_frames < 2 {
        return true;
    }
    let loop_start = loop_region.start_frames.clamp(0, total_frames as i64) as usize;
    let loop_end = loop_region
        .end_frames
        .clamp(loop_start as i64, total_frames as i64) as usize;
    if loop_end <= loop_start + 1 {
        return true;
    }

    let pre = compute_loop_boundary_metrics(data, channels, loop_start, loop_end);
    if pre.seam_score <= LOOP_REPAIR_GOOD_ENOUGH_SEAM_SCORE {
        eprintln!(
            "repair_skipped_good_enough: {context} loop=[{}..{}] seam={:.6} gap={:.6} local_jump={:.6}",
            loop_region.start_frames,
            loop_region.end_frames,
            pre.seam_score,
            pre.wrap_gap,
            pre.local_adjacent_jump,
        );
        return false;
    }
    let snapshot = data.to_vec();

    repair(data);

    let post = compute_loop_boundary_metrics(data, channels, loop_start, loop_end);
    let seam_improved = post.seam_score + 1.0e-12 < pre.seam_score;
    let jump_ok = post.local_adjacent_jump
        <= pre.local_adjacent_jump * LOOP_REPAIR_LOCAL_JUMP_REL_TOLERANCE
            + LOOP_REPAIR_LOCAL_JUMP_ABS_TOLERANCE;
    let accept = seam_improved && jump_ok;

    if !accept {
        data.copy_from_slice(&snapshot);
        eprintln!(
            "repair_reverted: {context} loop=[{}..{}] seam={:.6}->{:.6} gap={:.6}->{:.6} local_jump={:.6}->{:.6}",
            loop_region.start_frames,
            loop_region.end_frames,
            pre.seam_score,
            post.seam_score,
            pre.wrap_gap,
            post.wrap_gap,
            pre.local_adjacent_jump,
            post.local_adjacent_jump,
        );
    } else {
        eprintln!(
            "repair_applied: {context} loop=[{}..{}] seam={:.6}->{:.6} gap={:.6}->{:.6} local_jump={:.6}->{:.6}",
            loop_region.start_frames,
            loop_region.end_frames,
            pre.seam_score,
            post.seam_score,
            pre.wrap_gap,
            post.wrap_gap,
            pre.local_adjacent_jump,
            post.local_adjacent_jump,
        );
    }
    accept
}

/// Estimate the fundamental pitch period (in frames) of the loop body via autocorrelation.
/// Returns `None` if the loop body is too short, or if no clear periodic peak is found.
fn estimate_pitch_period_frames(
    data: &[f64],
    channels: usize,
    loop_start: usize,
    loop_end: usize,
) -> Option<f64> {
    if channels == 0 || loop_end <= loop_start {
        return None;
    }
    let loop_body = loop_end - loop_start;
    if loop_body < 128 {
        return None;
    }

    // Mono mixdown of loop body with DC removal.
    let mut mono = Vec::with_capacity(loop_body);
    let inv_ch = 1.0 / channels as f64;
    for f in loop_start..loop_end {
        let idx = f * channels;
        let mut s = 0.0;
        for ch in 0..channels {
            s += data[idx + ch];
        }
        mono.push(s * inv_ch);
    }
    let mean = mono.iter().sum::<f64>() / mono.len() as f64;
    for v in &mut mono {
        *v -= mean;
    }

    // Energy at lag 0 for normalization.
    let energy: f64 = mono.iter().map(|v| v * v).sum();
    if energy < 1.0e-12 {
        return None; // silence
    }

    // Autocorrelation for lags min_lag..=max_lag.
    // min_lag = ceil(sample_rate / 8000) ≈ 6 at 48kHz (highest expected fundamental)
    // max_lag = ceil(sample_rate / 30)   ≈ 1600 at 48kHz (lowest expected fundamental)
    let min_lag: usize = 6;
    let max_lag: usize = 1600.min(loop_body / 2);
    if max_lag <= min_lag {
        return None;
    }

    let mut acf = vec![0.0f64; max_lag + 1];
    for lag in min_lag..=max_lag {
        let mut sum = 0.0;
        let n = loop_body - lag;
        for i in 0..n {
            sum += mono[i] * mono[i + lag];
        }
        acf[lag] = sum / energy;
    }

    // Find first peak after first zero crossing (avoids octave errors).
    // Walk from min_lag until ACF goes negative, then find the first peak.
    let mut crossed_zero = false;
    let mut peak_lag: Option<usize> = None;
    for lag in min_lag..max_lag {
        if acf[lag] < 0.0 {
            crossed_zero = true;
        }
        if crossed_zero && acf[lag] > 0.0 && acf[lag] >= acf[lag + 1] && acf[lag] >= acf[lag - 1] {
            peak_lag = Some(lag);
            break;
        }
    }

    let lag = peak_lag?;
    if acf[lag] < 0.3 {
        return None; // weak correlation — aperiodic / noisy content
    }

    // Parabolic interpolation around the peak for sub-frame accuracy.
    let prev = if lag > 0 { acf[lag - 1] } else { acf[lag] };
    let next = if lag < max_lag {
        acf[lag + 1]
    } else {
        acf[lag]
    };
    let denom = 2.0 * (2.0 * acf[lag] - prev - next);
    let offset = if denom.abs() > 1.0e-12 {
        (prev - next) / denom
    } else {
        0.0
    };

    Some(lag as f64 + offset)
}

/// Compute the seam discontinuity at a forward loop wrap point.
/// Lower score = smoother loop. Measures the amplitude jump plus first and
/// second derivative discontinuities at the wrap (end-1 → start transition).
/// `amplitude_weight` scales the amplitude term, `slope_weight` scales the
/// derivative terms.
fn seam_score_at_forward_wrap(
    data: &[f64],
    channels: usize,
    loop_start_frame: usize,
    loop_end_frame: usize,
    amplitude_weight: f64,
    slope_weight: f64,
) -> f64 {
    if channels == 0 || loop_end_frame <= loop_start_frame + 1 {
        return 0.0;
    }
    if loop_end_frame < 2 {
        return 0.0;
    }
    let frames = sample_frame_count(data, channels) as usize;
    if loop_start_frame + 1 >= frames || loop_end_frame > frames {
        return 0.0;
    }

    let end_idx = (loop_end_frame - 1) * channels;
    let start_idx = loop_start_frame * channels;

    let mut score = 0.0;
    for ch in 0..channels {
        // C0: amplitude jump at the wrap point.
        let amp_jump = data[start_idx + ch] - data[end_idx + ch];
        score += amplitude_weight * amp_jump * amp_jump;

        // C1: first derivative (slope) discontinuity.
        if loop_end_frame >= 2 && loop_start_frame + 1 < frames {
            let slope_end = data[end_idx + ch] - data[(loop_end_frame - 2) * channels + ch];
            let slope_start = data[(loop_start_frame + 1) * channels + ch] - data[start_idx + ch];
            let slope_jump = slope_start - slope_end;
            score += slope_weight * slope_jump * slope_jump;
        }

        // C2: second derivative (curvature) discontinuity.
        if loop_end_frame >= 3 && loop_start_frame + 2 < frames {
            let curv_end = data[end_idx + ch] - 2.0 * data[(loop_end_frame - 2) * channels + ch]
                + data[(loop_end_frame - 3) * channels + ch];
            let curv_start = data[(loop_start_frame + 2) * channels + ch]
                - 2.0 * data[(loop_start_frame + 1) * channels + ch]
                + data[start_idx + ch];
            let curv_jump = curv_start - curv_end;
            score += slope_weight * 0.25 * curv_jump * curv_jump;
        }
    }

    score / channels as f64
}

/// Score ping-pong turnaround smoothness at both boundaries.
/// At a ping-pong turnaround the waveform should have slope ~0 (local extremum).
fn ping_pong_turnaround_cost(
    data: &[f64],
    channels: usize,
    start_frame: usize,
    end_frame: usize,
) -> f64 {
    let frames = data.len() / channels;
    if channels == 0 || frames < 3 || end_frame <= start_frame + 1 {
        return f64::MAX;
    }
    let mut cost = 0.0;

    // Start boundary: slope should be ~0
    if start_frame > 0 && start_frame + 1 < frames {
        for ch in 0..channels {
            let slope =
                data[(start_frame + 1) * channels + ch] - data[(start_frame - 1) * channels + ch];
            cost += slope * slope;
        }
    }

    // End boundary: slope at end-1 should be ~0
    if end_frame >= 2 && end_frame < frames {
        for ch in 0..channels {
            let slope =
                data[(end_frame - 1) * channels + ch] - data[(end_frame - 2) * channels + ch];
            // Also check continuity from end-1 to end (the reversal point)
            let slope2 = if end_frame < frames {
                data[end_frame * channels + ch] - data[(end_frame - 1) * channels + ch]
            } else {
                0.0
            };
            cost += slope * slope + slope2 * slope2;
        }
    }

    cost / channels as f64
}

/// Discovered loop points for an upscaled sample.
#[derive(Debug, Clone)]
pub struct DiscoveredLoopInfo {
    pub normal: SampleLoopRegion,
    pub sustain: SampleLoopRegion,
}

/// Search for optimal loop points near the approximate (proportionally-scaled) positions.
/// Uses pitch-period-aware search radius and a decoupled start/end sweep (O(3R) instead
/// of the old O(R²) brute force) to cover bass-register content without blowing up.
fn search_optimal_loop_points(
    data: &[f64],
    channels: usize,
    approx_start: i64,
    approx_end: i64,
    mode: SampleLoopMode,
) -> SampleLoopRegion {
    let total_frames = data.len() / channels;
    let loop_body = (approx_end - approx_start).max(1) as usize;

    // Very short loops: don't search, use proportional points.
    if loop_body < 4 || total_frames < 4 {
        return SampleLoopRegion {
            start_frames: approx_start.max(0),
            end_frames: approx_end.max(0).min(total_frames as i64),
            mode,
        };
    }

    // Pitch-aware search radius, capped at half the loop body so the search
    // can't drift into materially different content (e.g., a transient just
    // past the loop's natural end). AI phase drift is bounded by a few pitch
    // periods, not loop bodies, so this cap rarely binds on long loops but
    // prevents absurd drift on short ones.
    let pitch_period = estimate_pitch_period_frames(
        data,
        channels,
        approx_start.max(0) as usize,
        (approx_end.max(0) as usize).min(total_frames),
    );
    let max_radius = (loop_body / 2).clamp(4, 2048);
    let search_radius = match pitch_period {
        Some(period) => (period * 0.99).ceil() as usize,
        None => loop_body.div_euclid(8),
    }
    .clamp(4, max_radius);

    let start_min = (approx_start - search_radius as i64).max(0) as usize;
    let start_max =
        ((approx_start + search_radius as i64) as usize).min(total_frames.saturating_sub(2));

    if start_max < start_min {
        return SampleLoopRegion {
            start_frames: approx_start.max(0),
            end_frames: approx_end.max(0).min(total_frames as i64),
            mode,
        };
    }

    // Cost function that adds a gentle pitch-period alignment bias.
    let biased_cost = |raw_cost: f64, candidate: usize, proportional: i64| -> f64 {
        match pitch_period {
            Some(period) if period > 1.0 => {
                let dist = (candidate as f64 - proportional as f64).abs();
                let misalignment = (dist % period) / period;
                let penalty = misalignment.min(1.0 - misalignment); // 0.0 = on-period
                raw_cost + 0.5 * penalty
            }
            _ => raw_cost,
        }
    };

    // --- Fixed-length sweep: loop_end is always loop_start + predicted_length. ---
    // Preserves the original loop's period so the sustained tone keeps its pitch,
    // at the cost of not independently optimizing the end-side seam (the caller's
    // post-search crossfade handles any residual tail discontinuity).
    let predicted_length = loop_body; // approx_end - approx_start

    let proportional_start = approx_start.max(0) as usize;
    let proportional_end = (approx_end.max(0) as usize).min(total_frames);

    let mut best_start = proportional_start;
    let mut best_end = proportional_end;

    // Scoring closure for the current loop mode.
    let score_pair = |s: usize, e: usize| -> f64 {
        match mode {
            SampleLoopMode::Forward => seam_score_at_forward_wrap(data, channels, s, e, 4.0, 1.0),
            SampleLoopMode::PingPong => ping_pong_turnaround_cost(data, channels, s, e),
            SampleLoopMode::None => 0.0,
        }
    };

    if mode == SampleLoopMode::None {
        return SampleLoopRegion::none();
    }

    let mut best_cost = f64::MAX;
    for s in start_min..=start_max {
        let e = s + predicted_length;
        if e > total_frames {
            continue;
        }
        let cost = biased_cost(score_pair(s, e), s, approx_start);
        if cost < best_cost {
            best_cost = cost;
            best_start = s;
            best_end = e;
        }
    }

    // Fall back to proportional points if search didn't improve.
    let proportional_cost = biased_cost(score_pair(proportional_start, proportional_end), 0, 0);
    if best_cost >= proportional_cost {
        best_start = proportional_start;
        best_end = proportional_end;
    }

    SampleLoopRegion {
        start_frames: best_start as i64,
        end_frames: best_end as i64,
        mode,
    }
}

fn log_loop_search_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("QUINLIGHT_AUDIO_LOG_LOOPSEARCH")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

pub(crate) fn log_loop_search_result(
    label: &str,
    engine_name: &str,
    original_loop_info: SampleLoopInfo,
    original_rate: u32,
    target_rate: u32,
    discovered: &DiscoveredLoopInfo,
) {
    if !log_loop_search_enabled() {
        return;
    }
    let ms_per_frame = 1000.0 / target_rate as f64;
    for (kind, orig, disc) in [
        ("normal", original_loop_info.normal, discovered.normal),
        ("sustain", original_loop_info.sustain, discovered.sustain),
    ] {
        if orig.mode == SampleLoopMode::None {
            continue;
        }
        let predicted = scaled_loop_region(orig, original_rate, target_rate);
        let drift_start = disc.start_frames - predicted.start_frames;
        let drift_end = disc.end_frames - predicted.end_frames;
        let pred_len = (predicted.end_frames - predicted.start_frames).max(0);
        let disc_len = (disc.end_frames - disc.start_frames).max(0);
        let orig_len = (orig.end_frames - orig.start_frames).max(0);
        let drift_exceeds_loop = drift_start.abs() > pred_len || drift_end.abs() > pred_len;
        eprintln!(
            "[loopsearch] {label} engine={engine_name} loop={kind} \
             orig=[{}..{} len={}] @{}Hz -> \
             predicted=[{}..{} len={}] discovered=[{}..{} len={}] \
             drift=(start={:+} end={:+}) frames ({:+.2}ms / {:+.2}ms) \
             exceeds_loop={drift_exceeds_loop}",
            orig.start_frames,
            orig.end_frames,
            orig_len,
            original_rate,
            predicted.start_frames,
            predicted.end_frames,
            pred_len,
            disc.start_frames,
            disc.end_frames,
            disc_len,
            drift_start,
            drift_end,
            drift_start as f64 * ms_per_frame,
            drift_end as f64 * ms_per_frame,
        );
    }
}

/// Search for optimal loop points in upscaled audio for all loop types present.
pub(crate) fn search_all_loops(
    data: &[f64],
    channels: usize,
    original_loop_info: SampleLoopInfo,
    original_rate: u32,
    target_rate: u32,
) -> DiscoveredLoopInfo {
    let normal = if original_loop_info.normal.mode != SampleLoopMode::None
        && original_loop_info.normal.end_frames > original_loop_info.normal.start_frames
    {
        let approx = scaled_loop_region(original_loop_info.normal, original_rate, target_rate);
        search_optimal_loop_points(
            data,
            channels,
            approx.start_frames,
            approx.end_frames,
            original_loop_info.normal.mode,
        )
    } else {
        SampleLoopRegion::none()
    };

    let sustain = if original_loop_info.sustain.mode != SampleLoopMode::None
        && original_loop_info.sustain.end_frames > original_loop_info.sustain.start_frames
    {
        let approx = scaled_loop_region(original_loop_info.sustain, original_rate, target_rate);
        search_optimal_loop_points(
            data,
            channels,
            approx.start_frames,
            approx.end_frames,
            original_loop_info.sustain.mode,
        )
    } else {
        SampleLoopRegion::none()
    };

    DiscoveredLoopInfo { normal, sustain }
}

#[cfg(test)]
fn max_wrap_amplitude_gap(
    data: &[f64],
    channels: usize,
    loop_start_frame: usize,
    loop_end_frame: usize,
) -> f64 {
    if channels == 0 || loop_end_frame <= loop_start_frame {
        return 0.0;
    }
    let frames = sample_frame_count(data, channels) as usize;
    if loop_start_frame >= frames || loop_end_frame > frames {
        return 0.0;
    }
    let start_idx = loop_start_frame * channels;
    let end_idx = (loop_end_frame - 1) * channels;
    let mut max_gap = 0.0f64;
    for ch in 0..channels {
        max_gap = max_gap.max((data[start_idx + ch] - data[end_idx + ch]).abs());
    }
    max_gap
}

/// Clamp resampler/AI spikes: any adjacent jump larger than 1.5× the source
/// data's max loop-body jump is an artifact. Replace spiked frames with
/// linear interpolation from neighbors.
#[allow(dead_code)] // Used only by tests during pipeline refactor
fn clamp_resampler_spikes(
    data: &mut [f64],
    data_channels: usize,
    source_data: &[f64],
    source_channels: usize,
    source_loop_start: usize,
    source_loop_end: usize,
    context: &str,
) {
    if data_channels == 0 || data.is_empty() || source_channels == 0 || source_data.is_empty() {
        return;
    }
    let ls = source_loop_start;
    let le = source_loop_end;
    if le <= ls {
        return;
    }
    let src_frames = source_data.len() / source_channels;
    let mut src_max_jump = 0.0f64;
    for f in (ls + 1)..le.min(src_frames) {
        let idx = f * source_channels;
        let prev = (f - 1) * source_channels;
        if idx + source_channels <= source_data.len() {
            let j: f64 = (0..source_channels)
                .map(|ch| (source_data[idx + ch] - source_data[prev + ch]).abs())
                .fold(0.0f64, f64::max);
            src_max_jump = src_max_jump.max(j);
        }
    }
    let clamp_threshold = src_max_jump * 1.5;
    if clamp_threshold <= 0.0 {
        return;
    }
    let out_frames = data.len() / data_channels;
    for f in 1..out_frames.saturating_sub(1) {
        let idx = f * data_channels;
        let prev = (f - 1) * data_channels;
        let next = (f + 1) * data_channels;
        if next + data_channels > data.len() {
            break;
        }
        let jump: f64 = (0..data_channels)
            .map(|ch| (data[idx + ch] - data[prev + ch]).abs())
            .fold(0.0f64, f64::max);
        if jump > clamp_threshold {
            for ch in 0..data_channels {
                data[idx + ch] = (data[prev + ch] + data[next + ch]) * 0.5;
            }
            eprintln!(
                "  spike_clamped[{context}]: frame={f} jump={jump:.6} threshold={clamp_threshold:.6}"
            );
        }
    }
}

#[cfg(test)]
fn crossfade_loop_boundary(
    data: &mut [f64],
    channels: usize,
    loop_start_frames: i64,
    loop_end_frames: i64,
) -> f64 {
    crossfade_loop_boundary_ctx(data, channels, loop_start_frames, loop_end_frames, "")
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn crossfade_loop_boundary_ctx(
    data: &mut [f64],
    channels: usize,
    loop_start_frames: i64,
    loop_end_frames: i64,
    context: &str,
) -> f64 {
    if channels == 0 || data.is_empty() {
        return 0.0;
    }

    let frames = sample_frame_count(data, channels) as usize;
    let loop_start = loop_start_frames.clamp(0, frames as i64) as usize;
    let loop_end = loop_end_frames.clamp(loop_start as i64, frames as i64) as usize;
    if loop_end <= loop_start + 1 {
        return 0.0;
    }

    let end_idx = (loop_end - 1) * channels;
    let start_idx = loop_start * channels;

    // Measure gap and adjacent jumps BEFORE repair.
    let mut gap_before = 0.0f64;
    for ch in 0..channels {
        gap_before = gap_before.max((data[end_idx + ch] - data[start_idx + ch]).abs());
    }
    let adj_at_start = if loop_start > 0 {
        (0..channels)
            .map(|ch| (data[start_idx + ch] - data[(loop_start - 1) * channels + ch]).abs())
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };
    let adj_at_end = if loop_end >= 2 {
        (0..channels)
            .map(|ch| (data[end_idx + ch] - data[(loop_end - 2) * channels + ch]).abs())
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    // For nearly-seamless loops, skip crossfade — just force-match the endpoint.
    if gap_before < 0.01 {
        for ch in 0..channels {
            data[end_idx + ch] = data[start_idx + ch];
        }
        let adj_at_end_after = if loop_end >= 2 {
            (0..channels)
                .map(|ch| (data[end_idx + ch] - data[(loop_end - 2) * channels + ch]).abs())
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };
        eprintln!(
            "crossfade[{context}]: loop=[{loop_start}..{loop_end}] frames={frames} gap={gap_before:.6} (small, force-match only) adj_start={adj_at_start:.6} adj_end={adj_at_end:.6}->{adj_at_end_after:.6}"
        );
        return adj_at_end_after;
    }

    // Pitch-aware crossfade window: use one full pitch period when detectable,
    // so the blend covers a complete cycle and phase offsets rotate smoothly.
    let loop_len = loop_end - loop_start;
    let pitch_period = estimate_pitch_period_frames(data, channels, loop_start, loop_end);
    let window_frames = match pitch_period {
        Some(period) => (period.round() as usize).clamp(16, 2048).min(loop_len / 2),
        None => (loop_len / 4).clamp(1, 128),
    }
    .max(1);
    let head_samples: Vec<f64> =
        data[loop_start * channels..(loop_start + window_frames) * channels].to_vec();

    // Equal-power (sine-based) crossfade: preserves energy when head and tail
    // are out of phase, unlike linear blending which causes dips.
    let fade_start = loop_end - window_frames;
    for i in 0..window_frames {
        let t = (i + 1) as f64 / (window_frames + 1) as f64;
        let w_tail = (std::f64::consts::FRAC_PI_2 * (1.0 - t)).sin();
        let w_head = (std::f64::consts::FRAC_PI_2 * t).sin();
        let tail_idx = (fade_start + i) * channels;
        let head_idx = i * channels;
        for ch in 0..channels {
            data[tail_idx + ch] =
                data[tail_idx + ch] * w_tail + head_samples[head_idx + ch] * w_head;
        }
    }

    // Force exact endpoint continuity.
    for ch in 0..channels {
        data[end_idx + ch] = data[start_idx + ch];
    }

    // Measure gap AFTER repair.
    let gap_after: f64 = (0..channels)
        .map(|ch| (data[end_idx + ch] - data[start_idx + ch]).abs())
        .fold(0.0f64, f64::max);
    let adj_at_end_after = if loop_end >= 2 {
        (0..channels)
            .map(|ch| (data[end_idx + ch] - data[(loop_end - 2) * channels + ch]).abs())
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    // Print values around the wrap point.
    let vals_before_end = if loop_end >= 4 {
        format!(
            "[{e}-3]={:.4} [{e}-2]={:.4} [{e}-1]={:.4}",
            data[(loop_end - 3) * channels],
            data[(loop_end - 2) * channels],
            data[(loop_end - 1) * channels],
            e = loop_end
        )
    } else {
        String::new()
    };
    let vals_after_start = format!(
        "[{s}]={:.4} [{s}+1]={:.4} [{s}+2]={:.4}",
        data[loop_start * channels],
        data[(loop_start + 1) * channels],
        data[(loop_start + 2).min(loop_end - 1) * channels],
        s = loop_start
    );

    // Find the largest adjacent jump anywhere inside the loop body.
    let mut max_inner_jump = 0.0f64;
    let mut max_inner_jump_frame = loop_start;
    for f in (loop_start + 1)..loop_end {
        let idx = f * channels;
        let prev = (f - 1) * channels;
        if idx + channels <= data.len() {
            let jump: f64 = (0..channels)
                .map(|ch| (data[idx + ch] - data[prev + ch]).abs())
                .fold(0.0f64, f64::max);
            if jump > max_inner_jump {
                max_inner_jump = jump;
                max_inner_jump_frame = f;
            }
        }
    }

    eprintln!(
        "crossfade[{context}]: loop=[{loop_start}..{loop_end}] frames={frames} window={window_frames} gap={gap_before:.6}->{gap_after:.6} adj_start={adj_at_start:.6} adj_end={adj_at_end:.6}->{adj_at_end_after:.6} max_inner_jump={max_inner_jump:.6}@frame{max_inner_jump_frame} wrap: {vals_before_end} | {vals_after_start}"
    );
    max_inner_jump
}

fn frame_values(data: &[f64], channels: usize, frame_index: i64) -> Vec<f64> {
    if channels == 0 || frame_index < 0 {
        return Vec::new();
    }

    let frame_index = frame_index as usize;
    let sample_index = frame_index.saturating_mul(channels);
    if sample_index + channels > data.len() {
        return Vec::new();
    }

    data[sample_index..sample_index + channels].to_vec()
}

pub fn scaled_loop_region(
    loop_region: SampleLoopRegion,
    input_rate: u32,
    output_rate: u32,
) -> SampleLoopRegion {
    if !loop_region.has_loop() {
        return SampleLoopRegion::none();
    }

    SampleLoopRegion {
        start_frames: scaled_frame_count(
            loop_region.start_frames.max(0) as usize,
            input_rate,
            output_rate,
        ) as i64,
        end_frames: scaled_frame_count(
            loop_region.end_frames.max(0) as usize,
            input_rate,
            output_rate,
        ) as i64,
        mode: loop_region.mode,
    }
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn build_whole_sample_loop_input(data: &[f64], min_input_samples: usize) -> Vec<f64> {
    let align_to = min_input_samples.max(1);
    let min_total = min_input_samples.max(data.len() * 3);
    let aligned_len = ((min_total + align_to - 1) / align_to) * align_to;
    data.iter().copied().cycle().take(aligned_len).collect()
}

fn trim_sample_result(data: &[f64], channels: usize, keep_frames: i64) -> Vec<f64> {
    if channels == 0 || data.is_empty() {
        return Vec::new();
    }

    let keep_samples = (keep_frames.max(0) as usize).saturating_mul(channels);
    data[..data.len().min(keep_samples)].to_vec()
}

/// Which loop region the layered `[head][N×body][tail]` layout was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiledLoopSource {
    Normal,
    Sustain,
}

/// Layout metadata returned by `pad_for_engine` when the new
/// `[head][N×body][tail]` tiled layout is applied. `body_copies = N` tells the
/// post-engine extraction code where the middle body copy lives. The tail
/// length is not stored here — the extractor recovers it by subtracting head
/// and body from `SampleJob::original_length_48k_frames`, which absorbs the
/// rounding drift from multiple independent rate divisions.
#[derive(Debug, Clone, Copy)]
pub struct TilingLayout {
    pub body_copies: usize,
    pub loop_source: TiledLoopSource,
}

/// One tiled loop block inside the mixed-loop conditioning buffer. The block
/// has the `[head][body_copies × body][tail]` shape from the single-loop
/// tiling path. `offset_frames` is the starting position of the block in the
/// conditioning buffer (in conditioning-rate frames). The loop region and
/// mode come from the associated slot (`normal` / `sustain`) in
/// `SampleJob::loop_info` — no need to store them here.
#[derive(Debug, Clone, Copy)]
pub struct TiledBlock {
    pub offset_frames: i64,
    pub body_copies: usize,
}

/// Mixed-loop layout: the conditioning buffer carries a fixed-length base
/// timeline (the full source at conditioning rate) followed by two tiled
/// loop blocks — one per loop region — so the AI engine sees clean head→body
/// and body→tail transitions for both loops. All frame counts are at the
/// conditioning rate. The extracted output length stays equal to
/// `base_timeline_frames` (rescaled to the output rate); the tiled blocks are
/// conditioning-only context.
#[derive(Debug, Clone, Copy)]
pub struct MixedTilingLayout {
    pub base_timeline_frames: i64,
    pub sustain_block: TiledBlock,
    pub normal_block: TiledBlock,
}

/// Non-looped tiling layout: the full waveform is repeated `copies` times
/// back-to-back in the conditioning buffer, giving the AI engine N
/// independent passes at the same content. Post-extraction picks the
/// best-matching copy via FFT cross-correlation. `copy_frames` is the
/// length of one copy at the conditioning rate.
#[derive(Debug, Clone, Copy)]
pub struct RepeatedLayout {
    pub copies: usize,
    pub copy_frames: usize,
}

/// High-level engine-input layout. `None` on `SampleJob::engine_input_layout`
/// means the conditioning buffer was passed through unchanged (sample was
/// already long enough).
#[derive(Debug, Clone, Copy)]
pub enum EngineInputLayout {
    /// Single-loop `[head][N×body][tail]` layout.
    Single(TilingLayout),
    /// Mixed sustain + normal layout: base timeline + tiled sustain block +
    /// tiled normal block in one conditioning buffer.
    Mixed(MixedTilingLayout),
    /// Non-looped content repeated N times back-to-back.
    Repeated(RepeatedLayout),
}

impl EngineInputLayout {
    /// Extract the `TilingLayout` from a `Single` variant. Panics on other
    /// variants. Intended for tests; production code should pattern-match
    /// explicitly.
    #[cfg(test)]
    pub fn expect_single(&self) -> TilingLayout {
        match self {
            EngineInputLayout::Single(t) => *t,
            other => panic!("expected EngineInputLayout::Single, got {other:?}"),
        }
    }

    /// Extract the `MixedTilingLayout` from a `Mixed` variant. Panics on
    /// other variants. Intended for tests.
    #[cfg(test)]
    pub fn expect_mixed(&self) -> MixedTilingLayout {
        match self {
            EngineInputLayout::Mixed(m) => *m,
            other => panic!("expected EngineInputLayout::Mixed, got {other:?}"),
        }
    }

    /// Extract the `RepeatedLayout` from a `Repeated` variant. Panics on
    /// other variants. Intended for tests.
    #[cfg(test)]
    pub fn expect_repeated(&self) -> RepeatedLayout {
        match self {
            EngineInputLayout::Repeated(r) => *r,
            other => panic!("expected EngineInputLayout::Repeated, got {other:?}"),
        }
    }
}

/// Round `n` up to the next number of the form `4k + 1` (i.e. 5, 9, 13, 17,
/// …). PingPong layered layouts need this shape so that the middle body copy
/// (`body_copies / 2`, floor) is always a forward copy: with `N = 4k + 1`
/// copies alternating F,B,F,B,…,F, `N / 2 = 2k` is even, so the middle is an
/// F copy. Used only by `pad_for_engine`'s PingPong branch.
fn round_up_to_four_k_plus_one(n: usize) -> usize {
    let base = n.max(5);
    match base % 4 {
        0 => base + 1,
        1 => base,
        2 => base + 3,
        _ => base + 2,
    }
}

/// Snap a sample-index (channels-interleaved) to a frame boundary so tiling
/// code can't split a frame and cross-contaminate channels.
fn clamp_to_frame_samp(samp_index: usize, channels: usize) -> usize {
    (samp_index / channels.max(1)) * channels.max(1)
}

/// Pick the `body_copies` for a single tiled block so that
/// `head_samp + body_copies × loop_body_len + tail_samp ≥ min_samples`.
/// Forward loops get `max(3, n_from_min)`; ping-pong loops get the next
/// `4k + 1` (≥ 5) so the middle copy stays forward.
fn choose_body_copies(
    loop_body_len: usize,
    head_samp: usize,
    tail_samp: usize,
    min_samples: usize,
    mode: SampleLoopMode,
) -> usize {
    let non_body = head_samp.saturating_add(tail_samp);
    let need = min_samples.saturating_sub(non_body);
    let n_from_min = need.div_ceil(loop_body_len.max(1));
    if mode == SampleLoopMode::PingPong {
        round_up_to_four_k_plus_one(n_from_min)
    } else {
        n_from_min.max(3)
    }
}

/// Materialize a `[head] + body_copies × [body] + [tail]` tiled buffer for a
/// single loop region. `head = data[..loop_start_samp]`,
/// `tail = data[loop_end_samp..]`. Ping-pong alternates F,B,F,… with the
/// body reversed frame-by-frame on odd copies (channel order preserved
/// within each frame). Assumes `loop_start_samp ≤ loop_end_samp ≤ data.len()`
/// and all three are frame-boundary aligned.
fn materialize_tiled_block(
    data: &[f64],
    channels: usize,
    loop_start_samp: usize,
    loop_end_samp: usize,
    body_copies: usize,
    mode: SampleLoopMode,
) -> Vec<f64> {
    let loop_body_len = loop_end_samp.saturating_sub(loop_start_samp);
    let tail_samp = data.len().saturating_sub(loop_end_samp);
    let is_ping_pong = mode == SampleLoopMode::PingPong;
    let total_len = loop_start_samp
        .saturating_add(body_copies.saturating_mul(loop_body_len))
        .saturating_add(tail_samp);
    let body_frames = loop_body_len / channels.max(1);
    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&data[..loop_start_samp]);
    for k in 0..body_copies {
        if is_ping_pong && k % 2 == 1 {
            for frame in (0..body_frames).rev() {
                let frame_start = loop_start_samp + frame * channels;
                buf.extend_from_slice(&data[frame_start..frame_start + channels]);
            }
        } else {
            buf.extend_from_slice(&data[loop_start_samp..loop_end_samp]);
        }
    }
    if tail_samp > 0 {
        buf.extend_from_slice(&data[loop_end_samp..loop_end_samp + tail_samp]);
    }
    buf
}

/// Pad audio to meet engine minimum duration.
///
/// Returns a padded buffer plus layout metadata describing how the AI output
/// should be unpacked.
///
/// Layout selection (in priority order):
/// 1. **Mixed** (`EngineInputLayout::Mixed`) — both sustain and normal loops
///    are Forward/PingPong and normal is not fully contained in sustain.
///    Buffer shape: `[ base timeline ][ tiled sustain block ][ tiled normal
///    block ]`. Each tiled block has `[head][N×body][tail]` shape with the
///    same `body_copies` rules as the Single path. Extracted output length
///    stays at `base_timeline_frames` — the tiled blocks are conditioning
///    context only.
/// 2. **Single** (`EngineInputLayout::Single`) — exactly one valid loop
///    region (or normal fully contained in sustain → falls back to sustain).
///    Forward loops use the plain forward body for every copy (`N ≥ 3`).
///    PingPong loops alternate F,B,F,B,…,F, where B is the body reversed
///    end-to-start, and `N` is chosen as the smallest value of the form
///    `4k + 1` (≥ 5) that reaches `min_samples` — this guarantees the middle
///    copy is a forward body so post-engine extraction yields a sane loop
///    body with the stored sample's loop points still pointing at forward
///    audio. The head and tail are verbatim copies in both cases; the AI
///    sees a real head→F and F→tail transition regardless of mode.
/// 3. **None** — truly untiled pass-through. Non-looped samples already
///    meeting the minimum duration.
/// 4. **Repeated** — non-looped samples below the minimum duration are
///    tiled N times back-to-back, giving the AI engine independent passes
///    at the same content. Post-extraction picks the best copy via FFT
///    cross-correlation.
pub(crate) fn pad_for_engine(
    data: &[f64],
    channels: usize,
    min_samples: usize,
    loop_info: SampleLoopInfo,
) -> (Vec<f64>, Option<EngineInputLayout>) {
    if channels == 0 {
        return (data.to_vec(), None);
    }

    let normal_ok = loop_info.normal.has_loop()
        && matches!(
            loop_info.normal.mode,
            SampleLoopMode::Forward | SampleLoopMode::PingPong
        );
    let sustain_ok = loop_info.sustain.has_loop()
        && matches!(
            loop_info.sustain.mode,
            SampleLoopMode::Forward | SampleLoopMode::PingPong
        );

    // Mixed-loop path: both loops present, not full-containment.
    if normal_ok && sustain_ok && !data.is_empty() {
        let s = loop_info.sustain;
        let n = loop_info.normal;
        let normal_fully_in_sustain = s.start_frames <= n.start_frames
            && n.end_frames <= s.end_frames
            && s.start_frames < s.end_frames;
        if !normal_fully_in_sustain {
            return build_mixed_padded(data, channels, min_samples, s, n);
        }
        // Fall through to the single-sustain tiling path below.
    }

    // Single-loop dispatch. Reaching this point with both loops supported
    // means the mixed branch above fell through — i.e. normal ⊂ sustain
    // (full containment) — so prefer sustain in that case.
    let layered_loop = if normal_ok && sustain_ok {
        Some((TiledLoopSource::Sustain, loop_info.sustain))
    } else if normal_ok {
        Some((TiledLoopSource::Normal, loop_info.normal))
    } else if sustain_ok {
        Some((TiledLoopSource::Sustain, loop_info.sustain))
    } else {
        None
    };

    if let Some((loop_source, loop_region)) = layered_loop.filter(|_| !data.is_empty()) {
        // Clamp to data length, then snap back to a frame boundary. When the
        // loop region extends beyond `data`, the raw `.min(data.len())` can
        // leave a non-multiple-of-channels sample index, which would cause
        // the backward-copy loop below and stereo extend_from_slice to read
        // across frame boundaries and cross-contaminate L/R.
        let loop_start_samp = clamp_to_frame_samp(
            (loop_region.start_frames.max(0) as usize * channels).min(data.len()),
            channels,
        );
        let loop_end_samp = clamp_to_frame_samp(
            (loop_region.end_frames.max(0) as usize * channels).min(data.len()),
            channels,
        );
        let loop_body_len = loop_end_samp.saturating_sub(loop_start_samp);

        if loop_body_len == 0 {
            // Degenerate loop — fall through to non-looped path
            let mut padded = data.to_vec();
            padded.resize(min_samples, 0.0);
            return (padded, None);
        }

        let tail_samp = data.len().saturating_sub(loop_end_samp);
        let body_copies = choose_body_copies(
            loop_body_len,
            loop_start_samp,
            tail_samp,
            min_samples,
            loop_region.mode,
        );
        let padded = materialize_tiled_block(
            data,
            channels,
            loop_start_samp,
            loop_end_samp,
            body_copies,
            loop_region.mode,
        );

        return (
            padded,
            Some(EngineInputLayout::Single(TilingLayout {
                body_copies,
                loop_source,
            })),
        );
    }

    if data.len() >= min_samples {
        return (data.to_vec(), None);
    }

    // Non-looped: tile the full waveform N times back-to-back. This gives the
    // AI engine multiple independent passes at the same content, and
    // `extract_repeated_channel_output` picks the best copy via FFT
    // cross-correlation against reference_48k.
    //
    // All Forward/PingPong loops (normal or sustain) were handled by the
    // layered branches above; only truly non-looped samples shorter than
    // `min_samples` reach here.
    let total_frames = data.len() / channels;
    if total_frames == 0 {
        let padded = vec![0.0; min_samples];
        return (padded, None);
    }

    let copies = min_samples.div_ceil(data.len()).max(3);
    let mut padded = Vec::with_capacity(copies * data.len());
    for _ in 0..copies {
        padded.extend_from_slice(data);
    }

    (
        padded,
        Some(EngineInputLayout::Repeated(RepeatedLayout {
            copies,
            copy_frames: total_frames,
        })),
    )
}

/// Build a mixed-loop conditioning buffer:
///   `[ base timeline ][ tiled sustain block ][ tiled normal block ]`.
///
/// The base timeline is the full source data at the conditioning rate — no
/// truncation at sustain/normal end. Each tiled block has the same
/// `[head][N×body][tail]` shape as the single-loop tiling path, with its own
/// per-loop `body_copies`. Both blocks start from minimum copy counts
/// (forward = 3, ping-pong = 5); if the combined buffer is still shorter
/// than `min_samples`, the block whose loop body is larger grows first
/// (ping-pong in +4 increments so the middle copy stays forward). The AI
/// sees each loop's head→body and body→tail transitions directly, plus the
/// whole sample once through the base timeline.
fn build_mixed_padded(
    data: &[f64],
    channels: usize,
    min_samples: usize,
    sustain: SampleLoopRegion,
    normal: SampleLoopRegion,
) -> (Vec<f64>, Option<EngineInputLayout>) {
    let s_start = clamp_to_frame_samp(
        (sustain.start_frames.max(0) as usize * channels).min(data.len()),
        channels,
    );
    let s_end = clamp_to_frame_samp(
        (sustain.end_frames.max(0) as usize * channels).min(data.len()),
        channels,
    );
    let n_start = clamp_to_frame_samp(
        (normal.start_frames.max(0) as usize * channels).min(data.len()),
        channels,
    );
    let n_end = clamp_to_frame_samp(
        (normal.end_frames.max(0) as usize * channels).min(data.len()),
        channels,
    );

    let sustain_body_len = s_end.saturating_sub(s_start);
    let normal_body_len = n_end.saturating_sub(n_start);
    if sustain_body_len == 0 || normal_body_len == 0 {
        // Degenerate body on either side — let the caller fall back to the
        // non-looped / single-loop path.
        let mut padded = data.to_vec();
        padded.resize(min_samples, 0.0);
        return (padded, None);
    }

    // Compute minimum block lengths (copies = 3 forward, 5 ping-pong) first,
    // then grow to meet `min_samples` on the combined buffer.
    let base_len = data.len();
    let sustain_tail = base_len.saturating_sub(s_end);
    let normal_tail = base_len.saturating_sub(n_end);
    let sustain_is_pp = sustain.mode == SampleLoopMode::PingPong;
    let normal_is_pp = normal.mode == SampleLoopMode::PingPong;

    let block_len = |loop_start: usize, body: usize, tail: usize, copies: usize| -> usize {
        loop_start
            .saturating_add(copies.saturating_mul(body))
            .saturating_add(tail)
    };

    let mut sustain_copies = if sustain_is_pp { 5 } else { 3 };
    let mut normal_copies = if normal_is_pp { 5 } else { 3 };
    // Grow until `base_len + sustain_block + normal_block >= min_samples`.
    // Prefer growing the block whose loop body is larger so each added copy
    // buys more length. Ping-pong grows in +4 steps to preserve the 4k+1
    // invariant and keep the middle copy forward.
    loop {
        let total = base_len
            + block_len(s_start, sustain_body_len, sustain_tail, sustain_copies)
            + block_len(n_start, normal_body_len, normal_tail, normal_copies);
        if total >= min_samples {
            break;
        }
        if sustain_body_len >= normal_body_len {
            sustain_copies += if sustain_is_pp { 4 } else { 1 };
        } else {
            normal_copies += if normal_is_pp { 4 } else { 1 };
        }
    }

    let sustain_block_buf =
        materialize_tiled_block(data, channels, s_start, s_end, sustain_copies, sustain.mode);
    let normal_block_buf =
        materialize_tiled_block(data, channels, n_start, n_end, normal_copies, normal.mode);

    let base_timeline_frames = (base_len / channels.max(1)) as i64;
    let sustain_offset_samp = base_len;
    let normal_offset_samp = base_len + sustain_block_buf.len();

    let mut padded =
        Vec::with_capacity(base_len + sustain_block_buf.len() + normal_block_buf.len());
    padded.extend_from_slice(data);
    padded.extend_from_slice(&sustain_block_buf);
    padded.extend_from_slice(&normal_block_buf);

    let layout = EngineInputLayout::Mixed(MixedTilingLayout {
        base_timeline_frames,
        sustain_block: TiledBlock {
            offset_frames: (sustain_offset_samp / channels.max(1)) as i64,
            body_copies: sustain_copies,
        },
        normal_block: TiledBlock {
            offset_frames: (normal_offset_samp / channels.max(1)) as i64,
            body_copies: normal_copies,
        },
    });

    (padded, Some(layout))
}

/// Remove per-channel DC bias (mean subtraction) in place. SIMD fast paths for
/// mono and stereo; scalar striding for higher channel counts. Any sub-frame
/// remainder (data.len() not a multiple of channels) is handled by attributing
/// orphan samples to channels 0..remainder in order and subtracting the matching
/// channel mean, so no sample is left biased.
pub(crate) fn remove_dc_per_channel(data: &mut [f64], channels: usize) {
    if channels == 0 || data.is_empty() {
        return;
    }
    let frames = data.len() / channels;
    if frames == 0 {
        return;
    }
    let aligned_len = frames * channels;
    match channels {
        1 => {
            // aligned_len == data.len() for mono; no orphan possible.
            let mean = simd::sum_f64(data) / frames as f64;
            simd::subtract_in_place_f64(data, mean);
        }
        2 => {
            let (aligned, orphan) = data.split_at_mut(aligned_len);
            let (mut left, mut right) = simd::deinterleave_stereo_f64(aligned);
            let mean_l = simd::sum_f64(&left) / frames as f64;
            let mean_r = simd::sum_f64(&right) / frames as f64;
            simd::subtract_in_place_f64(&mut left, mean_l);
            simd::subtract_in_place_f64(&mut right, mean_r);
            let debiased = simd::interleave_stereo_f64(&left, &right);
            aligned[..debiased.len()].copy_from_slice(&debiased);
            // Orphan sample (at most one for stereo) is the left channel of an
            // incomplete trailing frame — debias it with mean_l.
            if let Some(s) = orphan.first_mut() {
                *s -= mean_l;
            }
        }
        _ => {
            // Compute each channel's mean from the frame-aligned prefix only,
            // then apply it to every sample belonging to that channel. Orphan
            // samples at index `aligned_len + k` belong to channel `k`, and the
            // step_by iterator naturally lands the matching mean on each one.
            for ch in 0..channels {
                let sum: f64 = data[..aligned_len]
                    .iter()
                    .skip(ch)
                    .step_by(channels)
                    .copied()
                    .sum();
                let mean = sum / frames as f64;
                for s in data.iter_mut().skip(ch).step_by(channels) {
                    *s -= mean;
                }
            }
        }
    }
}

/// RMS after per-channel DC removal, matching the normalization target to what
/// `normalize_sample()` computes on the engine output side.
pub(crate) fn dc_free_rms(data: &[f64], channels: usize) -> f64 {
    if data.is_empty() || channels == 0 {
        return 0.0;
    }
    let mut buf = data.to_vec();
    remove_dc_per_channel(&mut buf, channels);
    simd::rms_f64(&buf)
}

fn rms_or_zero(data: &[f64]) -> f64 {
    if data.is_empty() {
        0.0
    } else {
        simd::rms_f64(data)
    }
}

fn normalize_conditioning_channel(data: &mut [f64]) {
    if data.is_empty() {
        return;
    }

    let mean = (simd::sum_f64(data) / data.len() as f64) as f64;
    simd::subtract_in_place_f64(data, mean);

    let peak = simd::peak_abs_f64(data);
    if peak > 0.0 {
        simd::scale_in_place_f64(data, CONDITIONING_TARGET_PEAK / peak);
    }
}

pub(crate) fn normalize_conditioning_input(data: &mut [f64], channels: usize) {
    if channels == 0 || data.is_empty() {
        return;
    }

    let frames = data.len() / channels;
    if frames == 0 {
        return;
    }

    match channels {
        1 => normalize_conditioning_channel(data),
        2 => {
            let (mut left, mut right) = simd::deinterleave_stereo_f64(data);
            normalize_conditioning_channel(&mut left);
            normalize_conditioning_channel(&mut right);
            let normalized = simd::interleave_stereo_f64(&left, &right);
            data[..normalized.len()].copy_from_slice(&normalized);
        }
        _ => {
            for ch in 0..channels {
                let sum: f64 = data
                    .iter()
                    .skip(ch)
                    .step_by(channels)
                    .map(|&sample| sample as f64)
                    .sum();
                let mean = (sum / frames as f64) as f64;

                let mut peak = 0.0f64;
                for sample in data.iter_mut().skip(ch).step_by(channels) {
                    *sample -= mean;
                    peak = peak.max(sample.abs());
                }

                if peak > 0.0 {
                    let gain = CONDITIONING_TARGET_PEAK / peak;
                    for sample in data.iter_mut().skip(ch).step_by(channels) {
                        *sample *= gain;
                    }
                }
            }
        }
    }
}

fn build_canonical_segment_reference(
    saved_data: &[f64],
    original_rate: u32,
    channels: usize,
    loop_region: SampleLoopRegion,
    cleanup_settings: CleanupSettings,
) -> Result<Vec<f64>, String> {
    if channels == 0 || saved_data.is_empty() {
        return Ok(Vec::new());
    }

    let mut reference = if loop_region.has_loop() {
        let layout = build_boundary_loop_input(saved_data, channels, loop_region, saved_data.len());
        if cleanup_settings.mode == CleanupMode::Off {
            trim_sample_result(&layout, channels, sample_frame_count(saved_data, channels))
        } else {
            let cleaned = apply_cleanup(&layout, original_rate, channels, cleanup_settings)?;
            trim_sample_result(&cleaned, channels, sample_frame_count(saved_data, channels))
        }
    } else {
        apply_cleanup(saved_data, original_rate, channels, cleanup_settings)?
    };
    if reference.is_empty() {
        reference.shrink_to_fit();
    }
    Ok(reference)
}

fn build_retired_cleanup_segment_reference(
    saved_data: &[f64],
    original_rate: u32,
    channels: usize,
    loop_region: SampleLoopRegion,
    cleanup_preset: RetiredCleanupPreset,
) -> Result<Vec<f64>, String> {
    if channels == 0 || saved_data.is_empty() {
        return Ok(Vec::new());
    }

    let mut reference = if loop_region.has_loop() {
        let layout = build_boundary_loop_input(saved_data, channels, loop_region, saved_data.len());
        let cleaned =
            apply_retired_cleanup_preset(&layout, original_rate, channels, cleanup_preset)?;
        trim_sample_result(&cleaned, channels, sample_frame_count(saved_data, channels))
    } else {
        apply_retired_cleanup_preset(saved_data, original_rate, channels, cleanup_preset)?
    };

    if reference.is_empty() {
        reference.shrink_to_fit();
    }
    Ok(reference)
}

fn sustain_stitch_window_frames(
    sustain_end_frames: i64,
    saved_length_frames: i64,
    requested_window_frames: i64,
) -> i64 {
    requested_window_frames
        .max(0)
        .min(sustain_end_frames.max(0))
        .min((saved_length_frames - sustain_end_frames).max(0))
}

fn stitch_sustain_references(
    hold_reference: &[f64],
    release_reference: &[f64],
    channels: usize,
    sustain_end_frames: i64,
    stitch_window_frames: i64,
) -> Vec<f64> {
    if channels == 0 {
        return Vec::new();
    }

    let sustain_samples = (sustain_end_frames.max(0) as usize).saturating_mul(channels);
    let mut final_reference = release_reference.to_vec();
    let hold_copy_len = sustain_samples
        .min(hold_reference.len())
        .min(final_reference.len());
    final_reference[..hold_copy_len].copy_from_slice(&hold_reference[..hold_copy_len]);

    let saved_length_frames = sample_frame_count(&final_reference, channels);
    let window_frames = sustain_stitch_window_frames(
        sustain_end_frames,
        saved_length_frames,
        stitch_window_frames,
    );
    if window_frames <= 0 {
        return final_reference;
    }

    let window_samples = (window_frames as usize).saturating_mul(channels);
    let crossfade_start_frames = sustain_end_frames - window_frames;
    let crossfade_start_samples = (crossfade_start_frames.max(0) as usize).saturating_mul(channels);
    let crossfade_end_samples = crossfade_start_samples
        .saturating_add(window_samples)
        .min(hold_reference.len())
        .min(release_reference.len())
        .min(final_reference.len());
    let actual_window_samples = crossfade_end_samples.saturating_sub(crossfade_start_samples);
    if actual_window_samples == 0 {
        return final_reference;
    }

    for offset in 0..actual_window_samples {
        let t = (offset + channels) as f64 / (actual_window_samples + channels) as f64;
        final_reference[crossfade_start_samples + offset] =
            hold_reference[crossfade_start_samples + offset] * (1.0 - t)
                + release_reference[crossfade_start_samples + offset] * t;
    }
    final_reference
}

fn build_canonical_reference_parts_with_loop_prep(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    loop_prep: LoopPrepPlan,
    cleanup_settings: CleanupSettings,
) -> Result<LoopPreparedReference, String> {
    if channels == 0 || original.is_empty() {
        return Ok(LoopPreparedReference {
            final_reference: Vec::new(),
            hold_reference: None,
            release_reference: None,
        });
    }

    let saved_original = sample_prefix(original, channels, loop_prep.saved_length_frames);
    if saved_original.is_empty() {
        return Ok(LoopPreparedReference {
            final_reference: Vec::new(),
            hold_reference: None,
            release_reference: None,
        });
    }
    match loop_prep.mode {
        LoopPrepMode::BoundaryLoop => {
            let final_reference = build_canonical_segment_reference(
                saved_original,
                original_rate,
                channels,
                loop_prep.normal_loop,
                cleanup_settings,
            )?;
            // No pre-resample repair: the delta ramp modifies half the loop body,
            // creating a slope discontinuity that the SINC resampler amplifies
            // into a larger gap than the original. The post-resample crossfade
            // handles the boundary instead.
            Ok(LoopPreparedReference {
                final_reference,
                hold_reference: None,
                release_reference: None,
            })
        }
        LoopPrepMode::SustainPacked => {
            let hold_original =
                sample_prefix(saved_original, channels, loop_prep.sustain_loop.end_frames);
            let mut hold_reference = build_canonical_segment_reference(
                hold_original,
                original_rate,
                channels,
                loop_prep.sustain_loop,
                cleanup_settings,
            )?;
            repair_loop_body_in_place(&mut hold_reference, channels, loop_prep.sustain_loop);
            repair_attack_tail_to_loop_head_in_place(
                &mut hold_reference,
                channels,
                loop_prep.sustain_loop.start_frames,
            );

            let mut release_reference = build_canonical_segment_reference(
                saved_original,
                original_rate,
                channels,
                loop_prep.post_keyoff_loop(),
                cleanup_settings,
            )?;
            let hold_tail_anchor = frame_values(
                &hold_reference,
                channels,
                loop_prep.sustain_loop.end_frames - 1,
            );
            repair_release_head_to_loop_tail_in_place(
                &mut release_reference,
                &hold_tail_anchor,
                channels,
                loop_prep.sustain_loop.end_frames,
            );

            let mut sustain_aligned_release = stitch_sustain_references(
                &hold_reference,
                &release_reference,
                channels,
                loop_prep.sustain_loop.end_frames,
                loop_prep.sustain_loop.end_frames,
            );
            repair_loop_body_in_place(
                &mut sustain_aligned_release,
                channels,
                loop_prep.post_keyoff_loop(),
            );
            Ok(LoopPreparedReference {
                final_reference: sustain_aligned_release.clone(),
                hold_reference: Some(hold_reference),
                release_reference: Some(sustain_aligned_release),
            })
        }
        LoopPrepMode::WholeSampleLoopFallback => {
            eprintln!(
                "Loop prep fallback: using whole-sample periodic handling for degenerate metadata"
            );
            let mut final_reference = if cleanup_settings.mode == CleanupMode::Off {
                saved_original.to_vec()
            } else {
                let copy_len = saved_original.len();
                let mut tiled = Vec::with_capacity(copy_len * 3);
                tiled.extend_from_slice(saved_original);
                tiled.extend_from_slice(saved_original);
                tiled.extend_from_slice(saved_original);
                let cleaned = apply_cleanup(&tiled, original_rate, channels, cleanup_settings)?;
                cleaned[copy_len..copy_len * 2].to_vec()
            };
            let final_reference_frames = sample_frame_count(&final_reference, channels);
            repair_forward_loop_tail_in_place(
                &mut final_reference,
                channels,
                0,
                final_reference_frames,
            );
            Ok(LoopPreparedReference {
                final_reference,
                hold_reference: None,
                release_reference: None,
            })
        }
        LoopPrepMode::OneShot => {
            let final_reference = build_canonical_segment_reference(
                saved_original,
                original_rate,
                channels,
                SampleLoopRegion::none(),
                cleanup_settings,
            )?;
            Ok(LoopPreparedReference {
                final_reference,
                hold_reference: None,
                release_reference: None,
            })
        }
    }
}

fn build_canonical_reference_with_loop_prep(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    loop_prep: LoopPrepPlan,
    cleanup_settings: CleanupSettings,
) -> Result<Vec<f64>, String> {
    Ok(build_canonical_reference_parts_with_loop_prep(
        original,
        original_rate,
        channels,
        loop_prep,
        cleanup_settings,
    )?
    .final_reference)
}

#[allow(dead_code)]
fn build_canonical_reference(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    looped: bool,
    cleanup_settings: CleanupSettings,
) -> Result<Vec<f64>, String> {
    let loop_prep = LoopPrepPlan::from_loop_flag(sample_frame_count(original, channels), looped);
    build_canonical_reference_with_loop_prep(
        original,
        original_rate,
        channels,
        loop_prep,
        cleanup_settings,
    )
}

fn build_retired_cleanup_reference_with_loop_prep(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    loop_prep: LoopPrepPlan,
    cleanup_preset: RetiredCleanupPreset,
) -> Result<Vec<f64>, String> {
    if channels == 0 || original.is_empty() {
        return Ok(Vec::new());
    }

    let saved_original = sample_prefix(original, channels, loop_prep.saved_length_frames);
    if saved_original.is_empty() {
        return Ok(Vec::new());
    }

    match loop_prep.mode {
        LoopPrepMode::BoundaryLoop => {
            let final_reference = build_retired_cleanup_segment_reference(
                saved_original,
                original_rate,
                channels,
                loop_prep.normal_loop,
                cleanup_preset,
            )?;
            Ok(final_reference)
        }
        LoopPrepMode::SustainPacked => {
            let hold_original =
                sample_prefix(saved_original, channels, loop_prep.sustain_loop.end_frames);
            let mut hold_reference = build_retired_cleanup_segment_reference(
                hold_original,
                original_rate,
                channels,
                loop_prep.sustain_loop,
                cleanup_preset,
            )?;
            repair_loop_body_in_place(&mut hold_reference, channels, loop_prep.sustain_loop);
            repair_attack_tail_to_loop_head_in_place(
                &mut hold_reference,
                channels,
                loop_prep.sustain_loop.start_frames,
            );

            let mut release_reference = build_retired_cleanup_segment_reference(
                saved_original,
                original_rate,
                channels,
                loop_prep.post_keyoff_loop(),
                cleanup_preset,
            )?;
            let hold_tail_anchor = frame_values(
                &hold_reference,
                channels,
                loop_prep.sustain_loop.end_frames - 1,
            );
            repair_release_head_to_loop_tail_in_place(
                &mut release_reference,
                &hold_tail_anchor,
                channels,
                loop_prep.sustain_loop.end_frames,
            );

            let mut sustain_aligned_release = stitch_sustain_references(
                &hold_reference,
                &release_reference,
                channels,
                loop_prep.sustain_loop.end_frames,
                loop_prep.sustain_loop.end_frames,
            );
            repair_loop_body_in_place(
                &mut sustain_aligned_release,
                channels,
                loop_prep.post_keyoff_loop(),
            );
            Ok(sustain_aligned_release)
        }
        LoopPrepMode::WholeSampleLoopFallback => {
            let copy_len = saved_original.len();
            let mut tiled = Vec::with_capacity(copy_len * 3);
            tiled.extend_from_slice(saved_original);
            tiled.extend_from_slice(saved_original);
            tiled.extend_from_slice(saved_original);
            let cleaned =
                apply_retired_cleanup_preset(&tiled, original_rate, channels, cleanup_preset)?;
            let mut final_reference = cleaned[copy_len..copy_len * 2].to_vec();
            let final_reference_frames = sample_frame_count(&final_reference, channels);
            repair_forward_loop_tail_in_place(
                &mut final_reference,
                channels,
                0,
                final_reference_frames,
            );
            Ok(final_reference)
        }
        LoopPrepMode::OneShot => build_retired_cleanup_segment_reference(
            saved_original,
            original_rate,
            channels,
            SampleLoopRegion::none(),
            cleanup_preset,
        ),
    }
}

#[allow(dead_code)]
fn build_retired_cleanup_reference(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    looped: bool,
    cleanup_preset: RetiredCleanupPreset,
) -> Result<Vec<f64>, String> {
    let loop_prep = LoopPrepPlan::from_loop_flag(sample_frame_count(original, channels), looped);
    build_retired_cleanup_reference_with_loop_prep(
        original,
        original_rate,
        channels,
        loop_prep,
        cleanup_preset,
    )
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn build_one_shot_model_input(reference_data: &[f64], min_input_samples: usize) -> Vec<f64> {
    let align_to = min_input_samples.max(1);
    let aligned_len =
        ((reference_data.len().max(min_input_samples) + align_to - 1) / align_to) * align_to;
    let mut padded = reference_data.to_vec();
    padded.resize(aligned_len, 0.0);
    padded
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn build_segment_model_input(
    reference_data: &[f64],
    channels: usize,
    loop_region: SampleLoopRegion,
    min_input_samples: usize,
) -> Vec<f64> {
    if loop_region.has_loop() {
        build_boundary_loop_input(reference_data, channels, loop_region, min_input_samples)
    } else {
        build_one_shot_model_input(reference_data, min_input_samples)
    }
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn prepare_sample_data(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    loop_prep: LoopPrepPlan,
    cleanup_settings: CleanupSettings,
    min_duration_secs: f64,
    cancel_flag: &AtomicBool,
) -> Result<PreparedSampleData, String> {
    ensure_not_cancelled(cancel_flag)?;
    let reference_parts = build_canonical_reference_parts_with_loop_prep(
        original,
        original_rate,
        channels,
        loop_prep,
        cleanup_settings,
    )?;
    ensure_not_cancelled(cancel_flag)?;
    let target_rms = rms_or_zero(&reference_parts.final_reference);
    let min_input_samples = (min_duration_secs * original_rate as f64).ceil() as usize * channels;
    let (model_input, release_model_input, input_layout) = match loop_prep.mode {
        LoopPrepMode::SustainPacked => {
            let hold_reference = reference_parts
                .hold_reference
                .as_deref()
                .unwrap_or_default();
            let release_reference = reference_parts
                .release_reference
                .as_deref()
                .unwrap_or_default();
            let hold_model_input = build_segment_model_input(
                hold_reference,
                channels,
                loop_prep.sustain_loop,
                min_input_samples,
            );
            let release_model_input = build_segment_model_input(
                release_reference,
                channels,
                loop_prep.post_keyoff_loop(),
                min_input_samples,
            );
            let hold_input_frames = sample_frame_count(&hold_model_input, channels);
            let release_input_frames = sample_frame_count(&release_model_input, channels);
            (
                hold_model_input,
                Some(release_model_input),
                PreparedInputLayout::Sustain {
                    hold_input_frames,
                    release_input_frames,
                },
            )
        }
        LoopPrepMode::BoundaryLoop => {
            let model_input = build_segment_model_input(
                &reference_parts.final_reference,
                channels,
                loop_prep.normal_loop,
                min_input_samples,
            );
            let input_frames = sample_frame_count(&model_input, channels);
            (
                model_input,
                None,
                PreparedInputLayout::Single { input_frames },
            )
        }
        LoopPrepMode::WholeSampleLoopFallback => {
            let model_input =
                build_whole_sample_loop_input(&reference_parts.final_reference, min_input_samples);
            let input_frames = sample_frame_count(&model_input, channels);
            (
                model_input,
                None,
                PreparedInputLayout::Single { input_frames },
            )
        }
        LoopPrepMode::OneShot => {
            let model_input =
                build_one_shot_model_input(&reference_parts.final_reference, min_input_samples);
            let input_frames = sample_frame_count(&model_input, channels);
            (
                model_input,
                None,
                PreparedInputLayout::Single { input_frames },
            )
        }
    };
    ensure_not_cancelled(cancel_flag)?;
    let conditioning_input_48k = {
        let mut resampled = resample_audio_one_shot(&model_input, original_rate, 48_000, channels)?;
        normalize_conditioning_input(&mut resampled, channels);
        resampled
    };
    let (conditioning_input_48k_release, conditioning_input_48k_release_frames) =
        if let Some(release_input) = &release_model_input {
            let release = {
                let mut resampled =
                    resample_audio_one_shot(release_input, original_rate, 48_000, channels)?;
                normalize_conditioning_input(&mut resampled, channels);
                resampled
            };
            let frames = sample_frame_count(&release, channels);
            (Some(release), frames)
        } else {
            (None, 0)
        };
    ensure_not_cancelled(cancel_flag)?;
    Ok(PreparedSampleData {
        input_length_frames: sample_frame_count(&conditioning_input_48k, channels),
        reference_data: reference_parts.final_reference,
        hold_reference_data: reference_parts.hold_reference,
        release_reference_data: reference_parts.release_reference,
        model_input,
        release_model_input,
        conditioning_input_48k_frames: sample_frame_count(&conditioning_input_48k, channels),
        conditioning_input_48k,
        conditioning_input_48k_release,
        conditioning_input_48k_release_frames,
        input_layout,
        target_rms,
    })
}

#[cfg(test)]
fn resample_segment_reference(
    reference_data: &[f64],
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    loop_region: SampleLoopRegion,
    input_length_frames: Option<i64>,
) -> Result<Vec<f64>, String> {
    if !loop_region.has_loop() {
        return resample_audio_one_shot(reference_data, input_rate, output_rate, channels);
    }

    let keep_frames = sample_frame_count(reference_data, channels);
    let requested_min_frames = input_length_frames.unwrap_or(keep_frames).max(keep_frames);
    let mut layout = build_boundary_loop_input(
        reference_data,
        channels,
        loop_region,
        (requested_min_frames.max(0) as usize).saturating_mul(channels),
    );
    // Smooth the tiled junctions BEFORE resampling so the SINC kernel doesn't
    // ring at the discontinuities. Only modifies the "before" side of each
    // junction — the disposable tail of each copy. The "after" side (loop head)
    // is left untouched.
    smooth_tiled_loop_junctions(&mut layout, channels, keep_frames as usize, loop_region);
    // Diagnostic: scan tiled layout after smoothing for max jump.
    {
        let ls = loop_region.start_frames.max(0) as usize;
        let le = loop_region.end_frames.max(0) as usize;
        let layout_frames = layout.len() / channels.max(1);
        let mut max_j = 0.0f64;
        let mut max_j_frame = 0usize;
        for f in 1..layout_frames {
            let idx = f * channels;
            let prev = (f - 1) * channels;
            if idx + channels <= layout.len() {
                let j: f64 = (0..channels)
                    .map(|ch| (layout[idx + ch] - layout[prev + ch]).abs())
                    .fold(0.0f64, f64::max);
                if j > max_j {
                    max_j = j;
                    max_j_frame = f;
                }
            }
        }
        eprintln!(
            "  tiled_after_smooth: layout_frames={layout_frames} orig_loop=[{ls}..{le}] max_jump={max_j:.6}@frame{max_j_frame}"
        );
    }
    let mut resampled = resample_audio_one_shot(&layout, input_rate, output_rate, channels)?;
    clamp_resampler_spikes(
        &mut resampled,
        channels,
        reference_data,
        channels,
        loop_region.start_frames.max(0) as usize,
        loop_region.end_frames.max(0) as usize,
        "sinc_resample",
    );
    // Diagnostic: scan raw resampled output for max jump (before trim/crossfade).
    {
        let resamp_frames = resampled.len() / channels.max(1);
        let mut max_j = 0.0f64;
        let mut max_j_frame = 0usize;
        for f in 1..resamp_frames {
            let idx = f * channels;
            let prev = (f - 1) * channels;
            if idx + channels <= resampled.len() {
                let j: f64 = (0..channels)
                    .map(|ch| (resampled[idx + ch] - resampled[prev + ch]).abs())
                    .fold(0.0f64, f64::max);
                if j > max_j {
                    max_j = j;
                    max_j_frame = f;
                }
            }
        }
        eprintln!(
            "  resampled_raw: resamp_frames={resamp_frames} max_jump={max_j:.6}@frame{max_j_frame}"
        );
    }
    let output_keep_frames =
        scaled_frame_count(keep_frames.max(0) as usize, input_rate, output_rate);
    Ok(extract_middle_copy_from_loop_resample(
        &resampled,
        channels,
        output_keep_frames as i64,
    ))
}

#[cfg(test)]
fn extract_middle_copy_from_loop_resample(
    resampled: &[f64],
    channels: usize,
    keep_frames: i64,
) -> Vec<f64> {
    if channels == 0 || resampled.is_empty() {
        return Vec::new();
    }
    let keep_frames = keep_frames.max(0) as usize;
    if keep_frames == 0 {
        return Vec::new();
    }

    let total_frames = resampled.len() / channels;
    if total_frames <= keep_frames {
        return trim_sample_result(resampled, channels, keep_frames as i64);
    }

    // Prefer the center-most full-length copy. With odd copy counts this is
    // the true middle; with even counts it biases to the later of the two
    // central copies so the representative segment stays away from the attack.
    // If no aligned keep-sized copy fits, fall back to a centered window.
    //
    // Note: `middle_copy * keep_frames` only lands on a real copy boundary at
    // integer rate ratios. Tests using this helper set up integer-boundary
    // cases. Production extraction uses `tiled_offsets` (see `extract_result`),
    // which compensates for non-integer ratios via two-step `scaled_frame_count`
    // arithmetic.
    let full_copies = total_frames / keep_frames;
    if full_copies > 0 {
        let middle_copy = full_copies / 2;
        let start = middle_copy * keep_frames * channels;
        let end = start + keep_frames * channels;
        return resampled[start..end].to_vec();
    }

    let centered_start = (total_frames - keep_frames) / 2;
    let start = centered_start * channels;
    let end = start + keep_frames * channels;
    resampled[start..end].to_vec()
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn resample_reference_with_loop_prep(
    reference_data: &[f64],
    hold_reference_data: Option<&[f64]>,
    release_reference_data: Option<&[f64]>,
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    loop_prep: LoopPrepPlan,
    input_layout: Option<PreparedInputLayout>,
    sample_index: Option<i32>,
) -> Result<Vec<f64>, String> {
    if input_rate == output_rate {
        return Ok(reference_data.to_vec());
    }

    match loop_prep.mode {
        LoopPrepMode::BoundaryLoop => {
            let input_frames = match input_layout {
                Some(PreparedInputLayout::Single { input_frames }) => Some(input_frames),
                _ => None,
            };
            let mut resampled = resample_segment_reference(
                reference_data,
                input_rate,
                output_rate,
                channels,
                loop_prep.normal_loop,
                input_frames,
            )?;
            resampled = apply_source_frequency_blend_interleaved(
                &resampled,
                reference_data,
                input_rate,
                output_rate,
                channels,
                loop_prep.normal_loop.has_loop(),
            );
            // Diagnostic: scan the original-rate reference for the max inner jump.
            {
                let ls = loop_prep.normal_loop.start_frames.max(0) as usize;
                let le = loop_prep.normal_loop.end_frames.max(0) as usize;
                let ref_frames = sample_frame_count(reference_data, channels) as usize;
                let mut max_j = 0.0f64;
                let mut max_j_frame = ls;
                for f in (ls + 1)..le.min(ref_frames) {
                    let idx = f * channels;
                    let prev = (f - 1) * channels;
                    if idx + channels <= reference_data.len() {
                        let j: f64 = (0..channels)
                            .map(|ch| (reference_data[idx + ch] - reference_data[prev + ch]).abs())
                            .fold(0.0f64, f64::max);
                        if j > max_j {
                            max_j = j;
                            max_j_frame = f;
                        }
                    }
                }
                let sample_label = sample_index.map_or("?".to_string(), |i| format!("{}", i + 1));
                let orig_gap: f64 = if le > 0
                    && le * channels <= reference_data.len()
                    && ls * channels < reference_data.len()
                {
                    (0..channels)
                        .map(|ch| {
                            (reference_data[(le - 1) * channels + ch]
                                - reference_data[ls * channels + ch])
                                .abs()
                        })
                        .fold(0.0f64, f64::max)
                } else {
                    0.0
                };
                eprintln!(
                    "pre_resample[sample=#{sample_label}]: orig_loop=[{ls}..{le}] orig_gap={orig_gap:.6} orig_max_inner_jump={max_j:.6}@frame{max_j_frame} rate={input_rate}->{output_rate}"
                );
            }

            let scaled_loop = scaled_loop_region(loop_prep.normal_loop, input_rate, output_rate);
            // Smooth the attack-to-loop junction (modifies attack, not loop body).
            repair_attack_tail_to_loop_head_in_place(
                &mut resampled,
                channels,
                scaled_loop.start_frames,
            );
            // Smooth the loop wrap boundary (modifies loop tail toward loop head).
            let sample_label = sample_index.map_or("?".to_string(), |i| format!("{}", i + 1));
            let ctx = format!(
                "sinc_48k_ref sample=#{sample_label} orig_loop=[{}..{}]",
                loop_prep.normal_loop.start_frames, loop_prep.normal_loop.end_frames
            );
            let _ = crossfade_loop_boundary_ctx(
                &mut resampled,
                channels,
                scaled_loop.start_frames,
                scaled_loop.end_frames,
                &ctx,
            );
            Ok(resampled)
        }
        LoopPrepMode::SustainPacked => {
            let hold_input_frames = match input_layout {
                Some(PreparedInputLayout::Sustain {
                    hold_input_frames, ..
                }) => Some(hold_input_frames),
                _ => None,
            };
            let release_input_frames = match input_layout {
                Some(PreparedInputLayout::Sustain {
                    release_input_frames,
                    ..
                }) => Some(release_input_frames),
                _ => None,
            };
            let hold_reference = hold_reference_data.unwrap_or_default();
            let release_reference = release_reference_data.unwrap_or_default();
            let mut hold_resampled = resample_segment_reference(
                hold_reference,
                input_rate,
                output_rate,
                channels,
                loop_prep.sustain_loop,
                hold_input_frames,
            )?;
            hold_resampled = apply_source_frequency_blend_interleaved(
                &hold_resampled,
                hold_reference,
                input_rate,
                output_rate,
                channels,
                loop_prep.sustain_loop.has_loop(),
            );
            let scaled_sustain_loop =
                scaled_loop_region(loop_prep.sustain_loop, input_rate, output_rate);
            repair_loop_body_in_place(&mut hold_resampled, channels, scaled_sustain_loop);
            repair_attack_tail_to_loop_head_in_place(
                &mut hold_resampled,
                channels,
                scaled_sustain_loop.start_frames,
            );

            let mut release_resampled = resample_segment_reference(
                release_reference,
                input_rate,
                output_rate,
                channels,
                loop_prep.post_keyoff_loop(),
                release_input_frames,
            )?;
            release_resampled = apply_source_frequency_blend_interleaved(
                &release_resampled,
                release_reference,
                input_rate,
                output_rate,
                channels,
                loop_prep.post_keyoff_loop().has_loop(),
            );
            let scaled_sustain_end = scaled_frame_count(
                loop_prep.sustain_loop.end_frames as usize,
                input_rate,
                output_rate,
            ) as i64;
            let hold_tail_anchor = frame_values(&hold_resampled, channels, scaled_sustain_end - 1);
            repair_release_head_to_loop_tail_in_place(
                &mut release_resampled,
                &hold_tail_anchor,
                channels,
                scaled_sustain_end,
            );

            let scaled_saved_length = scaled_frame_count(
                loop_prep.saved_length_frames as usize,
                input_rate,
                output_rate,
            ) as i64;
            let scaled_window = scaled_sustain_end
                .min(scaled_saved_length - scaled_sustain_end)
                .max(0);
            let mut stitched = stitch_sustain_references(
                &hold_resampled,
                &release_resampled,
                channels,
                scaled_sustain_end,
                scaled_window,
            );
            let scaled_post_keyoff_loop =
                scaled_loop_region(loop_prep.post_keyoff_loop(), input_rate, output_rate);
            repair_loop_body_in_place(&mut stitched, channels, scaled_post_keyoff_loop);
            Ok(stitched)
        }
        LoopPrepMode::WholeSampleLoopFallback => {
            let mut resampled = resample_audio(
                reference_data,
                input_rate,
                output_rate,
                channels,
                if loop_prep.is_looped() {
                    ResampleBoundaryMode::LoopAware
                } else {
                    ResampleBoundaryMode::OneShot
                },
            )?;
            resampled = apply_source_frequency_blend_interleaved(
                &resampled,
                reference_data,
                input_rate,
                output_rate,
                channels,
                loop_prep.is_looped(),
            );
            if loop_prep.is_looped() {
                let resampled_frames = sample_frame_count(&resampled, channels);
                repair_forward_loop_tail_in_place(&mut resampled, channels, 0, resampled_frames);
            }
            Ok(resampled)
        }
        LoopPrepMode::OneShot => {
            let resampled =
                resample_audio_one_shot(reference_data, input_rate, output_rate, channels)?;
            Ok(apply_source_frequency_blend_interleaved(
                &resampled,
                reference_data,
                input_rate,
                output_rate,
                channels,
                false,
            ))
        }
    }
}

#[cfg(test)]
pub(crate) fn build_quinlight_reference_48k_with_loop_info(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    loop_info: SampleLoopInfo,
    cleanup_settings: CleanupSettings,
) -> Result<Vec<f64>, String> {
    build_quinlight_reference_48k_with_loop_info_indexed(
        original,
        original_rate,
        channels,
        loop_info,
        cleanup_settings,
        None,
    )
}

#[cfg(test)]
pub(crate) fn build_quinlight_reference_48k_with_loop_info_indexed(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    loop_info: SampleLoopInfo,
    cleanup_settings: CleanupSettings,
    sample_index: Option<i32>,
) -> Result<Vec<f64>, String> {
    let loop_prep = LoopPrepPlan::from_sample(sample_frame_count(original, channels), loop_info);
    let reference_parts = build_canonical_reference_parts_with_loop_prep(
        original,
        original_rate,
        channels,
        loop_prep,
        cleanup_settings,
    )?;
    resample_reference_with_loop_prep(
        &reference_parts.final_reference,
        reference_parts.hold_reference.as_deref(),
        reference_parts.release_reference.as_deref(),
        original_rate,
        48_000,
        channels,
        loop_prep,
        None,
        sample_index,
    )
}

#[cfg(test)]
pub(crate) fn build_quinlight_reference_48k(
    original: &[f64],
    original_rate: u32,
    channels: usize,
    looped: bool,
    cleanup_settings: CleanupSettings,
) -> Result<Vec<f64>, String> {
    let loop_info = if looped {
        SampleLoopInfo::forward(0, sample_frame_count(original, channels))
    } else {
        SampleLoopInfo::none()
    };
    build_quinlight_reference_48k_with_loop_info(
        original,
        original_rate,
        channels,
        loop_info,
        cleanup_settings,
    )
}

fn select_quinlight_mix_internal(
    reference_channels: i32,
    original_rate: u32,
    reference: QuinlightSelectionReference<'_>,
    engines: &[(String, Vec<f64>, i64, i32)],
    _engines_dispatched: usize,
    looped: bool,
) -> QuinlightMix {
    let source_reference = reference
        .source_native
        .filter(|reference| !reference.is_empty());
    let scoring_reference_48k = reference
        .score_48k
        .filter(|reference| !reference.is_empty());

    let mut scored: Vec<ScoredEngine<'_>> = engines
        .iter()
        .filter(|(_, data, _, channels)| {
            *channels == reference_channels && !data.is_empty() && reference_channels > 0
        })
        .map(|(name, data, length_frames, channels)| {
            let raw_score = if let Some(source_reference) = source_reference {
                crate::engine::spectral::spectral_correlation_across_rates(
                    source_reference,
                    original_rate,
                    data,
                    48_000,
                    *channels as usize,
                    looped,
                )
            } else if let Some(reference_48k) = scoring_reference_48k {
                crate::engine::spectral_correlation(
                    reference_48k,
                    data,
                    *channels as usize,
                    original_rate,
                )
            } else {
                0.0
            };
            ScoredEngine {
                name,
                data,
                length_frames: *length_frames,
                channels: *channels,
                raw_score,
            }
        })
        .collect();
    scored.sort_by(scored_engine_cmp);

    let floor = quinlight_usable_score_floor();

    let usable: Vec<ScoredEngine<'_>> = scored
        .iter()
        .filter(|engine| engine.raw_score >= floor)
        .cloned()
        .collect();

    let rejected: Vec<&ScoredEngine<'_>> = scored
        .iter()
        .filter(|engine| engine.raw_score < floor)
        .collect();
    if !rejected.is_empty() {
        let details = rejected
            .iter()
            .map(|e| format!("{} (score {:.4})", e.name, e.raw_score))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("  Quinlight: below floor {floor:.2} — {details}",);
    }

    // Consensus requires K >= 2 engines. A single engine has no cross-validation
    // and cannot attenuate hallucinations via spectral intersection.
    match usable.len() {
        0 | 1 => {
            if usable.len() == 1 {
                eprintln!(
                    "  Quinlight: only 1 usable engine ({}, score {:.4}); \
                     consensus requires 2+, keeping original sample",
                    usable[0].name, usable[0].raw_score,
                );
            } else {
                eprintln!(
                    "  Quinlight: no usable AI results (floor {:.2}); keeping original sample",
                    floor,
                );
            }
            let fallback_48k = reference.fallback_48k.unwrap_or(&[]);
            let reference_length_frames = if reference_channels > 0 {
                fallback_48k.len() as i64 / reference_channels as i64
            } else {
                0
            };
            QuinlightMix {
                data: fallback_48k.to_vec(),
                length_frames: reference_length_frames,
                channels: reference_channels,
                name: QUINLIGHT_NAME.to_string(),
                contributors: vec![QuinlightContributor {
                    name: QUINLIGHT_ORIGINAL_NAME.to_string(),
                    weight: 1.0,
                    score: 1.0,
                }],
            }
        }
        _ => spectral_intersection_blend(
            &usable,
            reference_channels,
            looped,
            source_reference,
            original_rate,
            reference.target_rms,
        ),
    }
}

#[cfg(test)]
pub(crate) fn select_quinlight_mix(
    reference_48k: &[f64],
    reference_channels: i32,
    original_rate: u32,
    engines: &[(String, Vec<f64>, i64, i32)],
    _engines_dispatched: usize,
    looped: bool,
) -> QuinlightMix {
    let target_rms = rms_or_zero(reference_48k);
    select_quinlight_mix_internal(
        reference_channels,
        original_rate,
        QuinlightSelectionReference {
            target_rms,
            source_native: None,
            score_48k: Some(reference_48k),
            fallback_48k: Some(reference_48k),
        },
        engines,
        _engines_dispatched,
        looped,
    )
}

/// Completed sample ready for live replacement.
#[derive(Clone)]
pub struct SampleResult {
    pub index: i32,
    pub data: Vec<f64>,
    pub length_frames: i64,
    pub channels: i32,
    pub sample_rate_hz: i32,
    /// Which engine produced this result.
    pub engine_name: String,
    /// Optimal loop points discovered in the upscaled audio (if sample has loops).
    pub discovered_loops: Option<DiscoveredLoopInfo>,
}

#[derive(Clone)]
pub enum RemasterOutput {
    Candidate(SampleResult),
    Final(SampleResult),
}

#[derive(Clone)]
struct ChannelResult {
    channel_index: usize,
    data: Vec<f64>,
    length_frames: i64,
}

#[derive(Default)]
struct PendingSampleOutputs {
    candidates: Vec<SampleResult>,
    engines_done: i32,
    engines_total: i32,
    // Size of `candidates` the last time we emitted a `Final` for this sample.
    // Lets us fire a progressive consensus when the 2nd candidate arrives and
    // re-fire when the 3rd refines the result, without emitting duplicates.
    last_emitted_candidate_count: usize,
}

type PendingOutputMap =
    std::sync::Arc<std::sync::Mutex<std::collections::HashMap<i32, PendingSampleOutputs>>>;

fn send_processing_update(
    progress_counter: &std::sync::atomic::AtomicI32,
    total_jobs: i32,
    progress_tx: &Sender<RemasterStatus>,
    sample_name: String,
) {
    use std::sync::atomic::Ordering;

    let current = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
    let _ = progress_tx.send(RemasterStatus::Processing {
        current,
        total: total_jobs,
        sample_name,
    });
}

fn send_engine_progress(
    progress_tx: &Sender<RemasterStatus>,
    sample_index: i32,
    engines_done: i32,
    num_engines: i32,
) {
    let _ = progress_tx.send(RemasterStatus::EngineProgress {
        sample_index,
        engines_done,
        engines_total: num_engines,
    });
}

fn conditioning_channel_name(source_channels: i32, channel_index: usize) -> &'static str {
    if source_channels == 2 {
        if channel_index == 0 { "left" } else { "right" }
    } else {
        "mono"
    }
}

fn conditioning_stem(sample_index: i32, channel_name: &str) -> String {
    match channel_name {
        "left" | "hold_left" | "release_left" => {
            let prefix = if channel_name.starts_with("hold_") {
                "hold_"
            } else if channel_name.starts_with("release_") {
                "release_"
            } else {
                ""
            };
            format!("sample_{sample_index}_{prefix}L")
        }
        "right" | "hold_right" | "release_right" => {
            let prefix = if channel_name.starts_with("hold_") {
                "hold_"
            } else if channel_name.starts_with("release_") {
                "release_"
            } else {
                ""
            };
            format!("sample_{sample_index}_{prefix}R")
        }
        _ => {
            let prefix = if channel_name.starts_with("hold_") {
                "hold_"
            } else if channel_name.starts_with("release_") {
                "release_"
            } else {
                ""
            };
            format!("sample_{sample_index}_{prefix}mono")
        }
    }
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn sustain_release_channel_name(source_channels: i32, channel_index: usize) -> &'static str {
    if source_channels == 2 {
        if channel_index == 0 {
            "release_left"
        } else {
            "release_right"
        }
    } else {
        "release_mono"
    }
}

#[allow(dead_code)] // Used only by tests during pipeline refactor
fn sustain_hold_channel_name(source_channels: i32, channel_index: usize) -> &'static str {
    if source_channels == 2 {
        if channel_index == 0 {
            "hold_left"
        } else {
            "hold_right"
        }
    } else {
        "hold_mono"
    }
}

fn engine_batch_item_for_input(job: &SampleJob, input: &PreparedChannelInput) -> EngineBatchItem {
    let original_rate_hz = job.rate.max(0) as u32;
    let source_stem = format!("sample_{}", job.index);
    EngineBatchItem {
        stem: conditioning_stem(job.index, &input.channel_name),
        sample_index: job.index,
        source_stem,
        conditioning_wav_path: input.input_path.to_string_lossy().into_owned(),
        original_rate_hz,
        original_nyquist_hz: original_nyquist_hz(original_rate_hz),
        conditioning_rate_hz: job.engine_input_rate(),
        conditioning_lowpass_hz: conditioning_lowpass_hz(original_rate_hz),
        source_channels: job.channels,
        conditioning_channels: if job.channels == 2 { 1 } else { job.channels },
        channel_index: input.channel_index,
        channel_name: input.channel_name.clone(),
        looped: job.looped,
        source_length_frames: job.source_length_frames,
        effective_length_frames: job.original_length_48k_frames,
    }
}

fn write_engine_batch_manifest(path: &Path, items: Vec<EngineBatchItem>) -> Result<(), String> {
    let manifest = EngineBatchManifest { version: 1, items };
    let json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("Failed to serialize engine manifest: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("Failed to write engine manifest: {e}"))
}

#[allow(dead_code)]
fn loop_prep_for_job(job: &SampleJob) -> LoopPrepPlan {
    LoopPrepPlan::from_sample(job.source_length_frames, job.loop_info)
}

#[cfg(test)]
fn reference_48k_from_job(job: &SampleJob) -> Result<Vec<f64>, String> {
    if job.channels <= 0 || job.reference_48k.is_empty() {
        return Ok(Vec::new());
    }
    Ok(job.reference_48k.clone())
}

fn build_quinlight_result(
    job: &SampleJob,
    candidates: &[SampleResult],
    engines_dispatched: usize,
) -> Result<SampleResult, String> {
    let engine_candidates: Vec<(String, Vec<f64>, i64, i32)> = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.engine_name.clone(),
                candidate.data.clone(),
                candidate.length_frames,
                candidate.channels,
            )
        })
        .collect();
    let reference_48k = &job.reference_48k;
    let mix = select_quinlight_mix_internal(
        job.channels,
        job.rate as u32,
        QuinlightSelectionReference {
            target_rms: job.target_rms,
            source_native: Some(&job.original_data),
            score_48k: if reference_48k.is_empty() {
                None
            } else {
                Some(reference_48k)
            },
            fallback_48k: if reference_48k.is_empty() {
                None
            } else {
                Some(reference_48k)
            },
        },
        &engine_candidates,
        engines_dispatched,
        job.looped,
    );
    if is_no_consensus_result(&mix.name) {
        // Prefer the 48 kHz SINC reference (already built upstream for spectral
        // scoring) so the upsample CLI can write a SINC fallback file. Module
        // batch callers filter this result via `should_apply_final_result`
        // regardless of data/rate, so this doesn't change what hits the tracker
        // pipeline. Fall through to the native original only if the SINC
        // reference was unavailable (e.g., empty source).
        let (data, length_frames, channels, sample_rate_hz) =
            if !mix.data.is_empty() && mix.channels > 0 {
                (mix.data, mix.length_frames, mix.channels, 48_000)
            } else {
                (
                    job.original_data.clone(),
                    job.source_length_frames,
                    job.channels,
                    job.rate,
                )
            };
        return Ok(SampleResult {
            index: job.index,
            data,
            length_frames,
            channels,
            sample_rate_hz,
            engine_name: mix.name,
            discovered_loops: None,
        });
    }
    let mut data = mix.data;
    let channels = mix.channels.max(1) as usize;

    // Search for optimal loop points in the blended result.
    let discovered_loops = if job.looped {
        let discovered = search_all_loops(&data, channels, job.loop_info, job.rate as u32, 48_000);
        log_loop_search_result(
            &job.display_name(),
            &mix.name,
            job.loop_info,
            job.rate as u32,
            48_000,
            &discovered,
        );
        Some(discovered)
    } else {
        None
    };

    // Pre-crossfade quality gate: compare the AI output's loop boundary against
    // the original sample's boundary. The original is the gold standard the artist
    // accepted. If AI makes it significantly worse, reject. Both amplitude gap and
    // slope mismatch are checked; slope is rate-normalized so the comparison is
    // fair across sample rates.
    if let Some(ref loops) = discovered_loops {
        if loops.normal.has_loop() {
            let ai_ls = loops.normal.start_frames.max(0) as usize;
            let ai_le = (loops.normal.end_frames.max(0) as usize).min(data.len() / channels);
            let orig_normal = &job.loop_info.normal;
            let orig_ch = (job.channels.max(1)) as usize;
            let orig_ls = orig_normal.start_frames.max(0) as usize;
            let orig_le =
                (orig_normal.end_frames.max(0) as usize).min(job.original_data.len() / orig_ch);

            if ai_le > ai_ls + 1 && orig_le > orig_ls + 1 {
                let ai_gap = loop_wrap_gap(&data, channels, ai_ls, ai_le);
                let ai_slope = loop_wrap_slope_gap(&data, channels, ai_ls, ai_le);

                let orig_gap = loop_wrap_gap(&job.original_data, orig_ch, orig_ls, orig_le);
                let orig_slope = loop_wrap_slope_gap(&job.original_data, orig_ch, orig_ls, orig_le);

                // Normalize AI slope to original's sample rate for fair comparison.
                // Slope per sample scales as 1/sample_rate, so multiply AI slope
                // by (48000 / orig_rate) to put both in the same units.
                let rate_scale = 48_000.0 / (job.rate.max(1) as f64);
                let ai_slope_normalized = ai_slope * rate_scale;

                // Allow up to AI_LOOP_SEAM_GATE_TOLERANCE times the original's
                // values, with absolute floors for near-perfect originals.
                let gap_limit =
                    (orig_gap * AI_LOOP_SEAM_GATE_TOLERANCE).max(AI_LOOP_SEAM_GATE_FLOOR);
                let slope_limit =
                    (orig_slope * AI_LOOP_SEAM_GATE_TOLERANCE).max(AI_LOOP_SEAM_SLOPE_FLOOR);

                // Loop-body slope envelope: sweep the loop interior window-by-
                // window and reject if the AI's per-window peak |slope|
                // materially exceeds the rate-normalized original's at the same
                // proportional position. Catches ticks/chirps/burst-capture
                // that the wrap-point check can't see.
                let ai_body_frames = ai_le - ai_ls;
                let num_windows = (ai_body_frames / AI_LOOP_BODY_WINDOW_FRAMES).max(1);
                let ai_env = loop_body_slope_envelope(&data, channels, ai_ls, ai_le, num_windows);
                let orig_env = loop_body_slope_envelope(
                    &job.original_data,
                    orig_ch,
                    orig_ls,
                    orig_le,
                    num_windows,
                );
                let mut body_reject_idx: Option<usize> = None;
                let mut body_reject_ai = 0.0f64;
                let mut body_reject_orig_norm = 0.0f64;
                let mut body_reject_limit = 0.0f64;
                if ai_env.len() == num_windows && orig_env.len() == num_windows {
                    // Verbose mode logs every window (not just the first
                    // rejection), producing `num_windows` lines per sample
                    // — can reach thousands on long-bodied loops. Diagnostic
                    // only; production runs leave `QUINLIGHT_AUDIO_LOG_LOOPSEARCH`
                    // unset.
                    let verbose = log_loop_search_enabled();
                    for i in 0..num_windows {
                        let orig_norm = orig_env[i] * rate_scale;
                        let limit = (orig_norm * AI_LOOP_BODY_SLOPE_TOLERANCE)
                            .max(AI_LOOP_BODY_SLOPE_FLOOR);
                        let ratio = if orig_norm > 0.0 {
                            ai_env[i] / orig_norm
                        } else {
                            f64::INFINITY
                        };
                        if verbose {
                            eprintln!(
                                "[bodygate] sample #{} engine={} window={i}/{num_windows} \
                                 ai_slope={:.6} orig_slope_norm={:.6} limit={:.6} ratio={:.3}{}",
                                job.index + 1,
                                mix.name,
                                ai_env[i],
                                orig_norm,
                                limit,
                                ratio,
                                if ai_env[i] > limit { " EXCEEDS" } else { "" },
                            );
                        }
                        if ai_env[i] > limit && body_reject_idx.is_none() {
                            body_reject_idx = Some(i);
                            body_reject_ai = ai_env[i];
                            body_reject_orig_norm = orig_norm;
                            body_reject_limit = limit;
                            if !verbose {
                                break;
                            }
                        }
                    }
                }

                // HF-concentration gate: is the AI's added HF energy (above
                // orig Nyquist) disproportionately concentrated at the loop
                // boundary vs. the loop body's own interior baseline? A "yes"
                // indicates a tick/chirp/burst captured at the boundary that
                // the wrap and slope-envelope checks missed.
                //
                // We compare boundary-vs-middle *within the loop body* rather
                // than boundary-vs-whole-sample — sample-wide averages get
                // dragged down by silent attacks, which then falsely flags
                // legitimate AI brightening of the sustained loop body.
                let orig_nyquist_hz = (job.rate.max(1) as f64) * 0.5;
                // Per-frame peak |sample| across channels (not mean) so a tick
                // on either channel contributes fully to the HF envelope.
                // Consistent with wrap/body-slope gates, which also take the
                // max across channels rather than averaging.
                let ai_mono: Vec<f64> = if channels <= 1 {
                    data.clone()
                } else {
                    let frame_count = data.len() / channels;
                    let mut mono = Vec::with_capacity(frame_count);
                    for f in 0..frame_count {
                        let mut peak = 0.0f64;
                        for ch in 0..channels {
                            let v = data[f * channels + ch].abs();
                            if v > peak {
                                peak = v;
                            }
                        }
                        mono.push(peak);
                    }
                    mono
                };
                let hf_envelope =
                    crate::engine::spectral::hf_energy_envelope(&ai_mono, 48_000, orig_nyquist_hz);
                let mut hf_reject_info: Option<(f64, f64, f64)> = None; // (middle_avg, local_max, ratio)
                let r = AI_LOOP_HF_RADIUS_STFT_FRAMES;
                if hf_envelope.len() >= AI_LOOP_HF_MIN_BODY_STFT_FRAMES {
                    let ls_stft = crate::engine::spectral::stft_frame_for_sample(ai_ls, 48_000);
                    let le_stft = crate::engine::spectral::stft_frame_for_sample(
                        ai_le.saturating_sub(1),
                        48_000,
                    );
                    let env_len = hf_envelope.len();
                    // Middle of the loop body: strictly between the two boundary
                    // radii. Skip the check if the body isn't long enough to
                    // have a meaningful middle region.
                    let middle_start = (ls_stft + r + 1).min(env_len);
                    let middle_end = le_stft.saturating_sub(r).min(env_len);
                    if middle_end > middle_start
                        && middle_end - middle_start >= AI_LOOP_HF_MIN_MIDDLE_STFT_FRAMES
                    {
                        let middle_slice = &hf_envelope[middle_start..middle_end];
                        let middle_avg =
                            middle_slice.iter().sum::<f64>() / middle_slice.len() as f64;
                        if middle_avg >= AI_LOOP_HF_CONCENTRATION_FLOOR {
                            let mut local_hf_max = 0.0f64;
                            for center in [ls_stft, le_stft] {
                                let lo = center.saturating_sub(r);
                                let hi = (center + r + 1).min(env_len);
                                for v in &hf_envelope[lo..hi] {
                                    if *v > local_hf_max {
                                        local_hf_max = *v;
                                    }
                                }
                            }
                            let ratio = local_hf_max / middle_avg;
                            if log_loop_search_enabled() {
                                eprintln!(
                                    "[hfgate] sample #{} engine={} middle_avg={middle_avg:.6e} \
                                     local_hf_max={local_hf_max:.6e} ratio={ratio:.3} \
                                     middle=[{middle_start}..{middle_end}] ls_stft={ls_stft} \
                                     le_stft={le_stft} env_len={env_len}",
                                    job.index + 1,
                                    mix.name,
                                );
                            }
                            if ratio > AI_LOOP_HF_CONCENTRATION_TOLERANCE {
                                hf_reject_info = Some((middle_avg, local_hf_max, ratio));
                            }
                        }
                    } else if log_loop_search_enabled() {
                        eprintln!(
                            "[hfgate] sample #{} engine={} skipped: body too short \
                             (middle=[{middle_start}..{middle_end}] env_len={env_len})",
                            job.index + 1,
                            mix.name,
                        );
                    }
                }

                let reject = ai_gap > gap_limit
                    || ai_slope_normalized > slope_limit
                    || body_reject_idx.is_some()
                    || hf_reject_info.is_some();
                if reject {
                    if let Some((avg_hf, local_hf_max, ratio)) = hf_reject_info {
                        eprintln!(
                            "loop quality gate: sample #{} hf_concentration ratio={ratio:.3} \
                             (limit {:.3}) local_hf_max={local_hf_max:.6e} avg_hf={avg_hf:.6e} \
                             — keeping original",
                            job.index + 1,
                            AI_LOOP_HF_CONCENTRATION_TOLERANCE,
                        );
                    } else if let Some(idx) = body_reject_idx {
                        eprintln!(
                            "loop quality gate: sample #{} body_window={idx}/{num_windows} \
                             ai_slope={body_reject_ai:.6} (limit {body_reject_limit:.6}) \
                             orig_slope_norm={body_reject_orig_norm:.6} \
                             ai_gap={ai_gap:.6} ai_slope={ai_slope_normalized:.6} \
                             — keeping original",
                            job.index + 1,
                        );
                    } else {
                        eprintln!(
                            "loop quality gate: sample #{} ai_gap={ai_gap:.6} (limit {gap_limit:.6}) \
                             ai_slope={ai_slope_normalized:.6} (limit {slope_limit:.6}) \
                             orig_gap={orig_gap:.6} orig_slope={orig_slope:.6} — keeping original",
                            job.index + 1,
                        );
                    }
                    return Ok(SampleResult {
                        index: job.index,
                        data: job.original_data.clone(),
                        length_frames: job.source_length_frames,
                        channels: job.channels,
                        sample_rate_hz: job.rate,
                        engine_name: format!("{} (loop gate)", mix.name),
                        discovered_loops: None,
                    });
                }
            }
        }
    }

    // Crossfade the AI output at the discovered loop boundaries. The AI engine
    // doesn't know about loop semantics, so the waveform at loop_end may not
    // match loop_start. Without this, the wrap point produces an audible click.
    let mut post_xfade_max_jump: f64 = 0.0;
    if let Some(ref loops) = discovered_loops {
        if loops.normal.has_loop() {
            let j = crossfade_loop_boundary_ctx(
                &mut data,
                channels,
                loops.normal.start_frames,
                loops.normal.end_frames,
                "ai_output_normal",
            );
            post_xfade_max_jump = post_xfade_max_jump.max(j);
        }
        if loops.sustain.has_loop() {
            let j = crossfade_loop_boundary_ctx(
                &mut data,
                channels,
                loops.sustain.start_frames,
                loops.sustain.end_frames,
                "ai_output_sustain",
            );
            post_xfade_max_jump = post_xfade_max_jump.max(j);
        }
    }
    if post_xfade_max_jump > AI_LOOP_POST_CROSSFADE_INNER_JUMP_LIMIT {
        eprintln!(
            "loop quality gate (post-crossfade): sample #{} max_inner_jump={post_xfade_max_jump:.6} \
             (limit {:.6}) — keeping original",
            job.index + 1,
            AI_LOOP_POST_CROSSFADE_INNER_JUMP_LIMIT,
        );
        return Ok(SampleResult {
            index: job.index,
            data: job.original_data.clone(),
            length_frames: job.source_length_frames,
            channels: job.channels,
            sample_rate_hz: job.rate,
            engine_name: format!("{} (loop gate)", mix.name),
            discovered_loops: None,
        });
    }

    let length_frames = if channels > 0 {
        data.len() as i64 / channels as i64
    } else {
        mix.length_frames
    };
    Ok(SampleResult {
        index: job.index,
        data,
        length_frames,
        channels: mix.channels,
        sample_rate_hz: job.output_sample_rate_hz,
        engine_name: mix.name,
        discovered_loops,
    })
}

fn count_eligible_engines_for_job(
    job: &SampleJob,
    engines: &[Box<dyn engine::UpsampleEngine>],
) -> i32 {
    let original_rate_hz = job.rate.max(0) as u32;
    engines
        .iter()
        .filter(|engine| engine.supports_original_rate(original_rate_hz))
        .count() as i32
}

fn record_engine_completion(
    job: &SampleJob,
    candidate: Option<SampleResult>,
    progress_counter: &std::sync::atomic::AtomicI32,
    total_jobs: i32,
    progress_tx: &Sender<RemasterStatus>,
    result_tx: &Sender<RemasterOutput>,
    pending_outputs: &PendingOutputMap,
    progress_message: String,
    eligible_engine_total: i32,
    cancel_flag: &AtomicBool,
    progressive: bool,
) {
    send_processing_update(progress_counter, total_jobs, progress_tx, progress_message);

    let candidate_for_send = candidate.clone();
    let (engines_done, final_candidates, engines_total) = {
        let mut pending = pending_outputs.lock().unwrap();
        let sample = pending.entry(job.index).or_default();
        sample.engines_total = sample.engines_total.max(eligible_engine_total);
        if let Some(candidate) = candidate {
            if let Some(existing) = sample
                .candidates
                .iter_mut()
                .find(|existing| existing.engine_name == candidate.engine_name)
            {
                *existing = candidate;
            } else {
                sample.candidates.push(candidate);
                sample.candidates.sort_by(|lhs, rhs| {
                    crate::engine::engine_preference_rank(&lhs.engine_name)
                        .cmp(&crate::engine::engine_preference_rank(&rhs.engine_name))
                        .then_with(|| lhs.engine_name.cmp(&rhs.engine_name))
                });
            }
        }
        sample.engines_done += 1;
        let engines_done = sample.engines_done;

        // Emit Final when new candidates have arrived. In progressive mode
        // (GUI) fire as soon as 2 candidates land and again when more grow
        // the set. In non-progressive mode (CLI) fire only once all eligible
        // engines have completed, so the spectral intersection runs exactly
        // once per sample. The "all engines done" path also covers the
        // single-candidate no-consensus fallback in both modes.
        let all_engines_done =
            sample.engines_done >= sample.engines_total && sample.engines_total > 0;
        let have_new_candidates = sample.candidates.len() > sample.last_emitted_candidate_count
            && !sample.candidates.is_empty();
        let progressive_ready = progressive && sample.candidates.len() >= 2 && have_new_candidates;
        let finalised = all_engines_done && have_new_candidates;
        let final_candidates = if progressive_ready || finalised {
            sample.last_emitted_candidate_count = sample.candidates.len();
            Some(sample.candidates.clone())
        } else {
            None
        };
        (engines_done, final_candidates, sample.engines_total)
    };

    send_engine_progress(progress_tx, job.index, engines_done, engines_total);

    if let Some(candidate) = candidate_for_send {
        let _ = result_tx.send(RemasterOutput::Candidate(candidate));
    }

    if cancellation_requested(cancel_flag) {
        return;
    }

    if let Some(final_candidates) = final_candidates {
        match build_quinlight_result(job, &final_candidates, engines_total as usize) {
            Ok(final_result) => {
                let _ = result_tx.send(RemasterOutput::Final(final_result));
            }
            Err(e) => {
                eprintln!(
                    "Quinlight Audio finalize failed for {}: {e}",
                    job.display_name()
                );
            }
        }
    }
}

fn store_channel_result(
    pending_channels: &mut std::collections::HashMap<usize, Vec<ChannelResult>>,
    job_idx: usize,
    channel_result: ChannelResult,
) {
    let entry = pending_channels.entry(job_idx).or_default();
    if let Some(existing) = entry
        .iter_mut()
        .find(|existing| existing.channel_index == channel_result.channel_index)
    {
        *existing = channel_result;
    } else {
        entry.push(channel_result);
        entry.sort_by_key(|result| result.channel_index);
    }
}

fn maybe_finish_engine_job(
    job_idx: usize,
    jobs: &[SampleJob],
    engine_name: &str,
    engine_cache_id: &str,
    ddim_steps: u16,
    engine_label: &str,
    progress_counter: &std::sync::atomic::AtomicI32,
    total_jobs: i32,
    progress_tx: &Sender<RemasterStatus>,
    result_tx: &Sender<RemasterOutput>,
    processed: &mut std::collections::HashSet<usize>,
    pending_outputs: &PendingOutputMap,
    pending_channels: &mut std::collections::HashMap<usize, Vec<ChannelResult>>,
    eligible_engine_counts_by_job: &[i32],
    success_counter: &std::sync::atomic::AtomicI32,
    cancel_flag: &AtomicBool,
    progressive: bool,
) -> Result<bool, String> {
    if processed.contains(&job_idx) {
        return Ok(false);
    }

    let job = &jobs[job_idx];
    let expected_channels = job.conditioning_inputs.len();
    let ready = pending_channels
        .get(&job_idx)
        .is_some_and(|results| results.len() == expected_channels);
    if !ready {
        return Ok(false);
    }

    let channel_results = pending_channels.remove(&job_idx).unwrap_or_default();
    let mut result = assemble_sample_candidate(job, engine_name, &channel_results)?;
    cache_store(
        &job.pcm_sha256,
        engine_cache_id,
        ddim_steps,
        &result,
        job.target_rms,
    );
    normalize_sample(&mut result, job.target_rms);
    record_engine_completion(
        job,
        Some(result),
        progress_counter,
        total_jobs,
        progress_tx,
        result_tx,
        pending_outputs,
        format!("{}: {}", engine_label, job.display_name()),
        eligible_engine_counts_by_job[job_idx],
        cancel_flag,
        progressive,
    );
    success_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    processed.insert(job_idx);
    Ok(true)
}

/// Pre-extracted sample job for AudioSR processing.
#[derive(Clone)]
#[allow(dead_code)]
pub struct SampleJob {
    pub index: i32,
    pub name: String,
    /// Untouched original waveform used when Quinlight keeps the sample unchanged.
    pub original_data: Vec<f64>,
    /// Original sample rate of the source sample.
    pub rate: i32,
    pub output_sample_rate_hz: i32,
    pub channels: i32,
    /// Source format bit depth (e.g. 8, 16). Used in cache key per spec.
    pub bits_per_sample: u8,
    pub source_length_frames: i64,
    pub looped: bool,
    pub loop_info: SampleLoopInfo,
    /// Conditioning WAVs passed to AI engines (at `conditioning_rate_hz`).
    /// Stereo samples are split into independent mono channel jobs.
    pub conditioning_inputs: Vec<PreparedChannelInput>,
    /// Sample rate of the conditioning WAVs (24kHz or 48kHz).
    pub conditioning_rate_hz: u32,
    /// Inner SHA-256 of the prepared PCM bytes (engine-independent).
    /// The full per-engine cache key is computed at lookup/store time.
    pub pcm_sha256: [u8; 32],
    pub target_rms: f64,
    /// Raw original resampled to 48kHz for spectral scoring.
    pub reference_48k: Vec<f64>,
    /// Expected output length in frames at 48kHz (before padding).
    pub original_length_48k_frames: i64,
    /// Engine-input layout metadata. `Single` describes the classic
    /// `[head][N×body][tail]` tiled buffer; `Mixed` describes the
    /// `[base timeline][tiled sustain block][tiled normal block]` buffer
    /// used when a sample has both sustain and normal loops. `None` means
    /// the conditioning buffer was passed through unchanged (non-looped /
    /// already long enough, or a non-layered fallback path).
    pub engine_input_layout: Option<EngineInputLayout>,
    /// Optional native-rate WAV bundle consumed by engines whose Python
    /// wrappers resample internally (AudioSR, LavaSR, FLowHigh, AP-BWE).
    /// Populated only when `native_rate != conditioning_rate`; otherwise the
    /// conditioning WAV already is at native rate and is reused. Empty /
    /// `0` / `None` means "fall back to conditioning_*".
    pub native_inputs: Vec<PreparedChannelInput>,
    pub native_rate_hz: u32,
    pub native_input_layout: Option<EngineInputLayout>,
}

impl SampleJob {
    fn has_native_bundle(&self) -> bool {
        self.native_rate_hz != 0 && !self.native_inputs.is_empty()
    }

    pub fn engine_input_rate(&self) -> u32 {
        if self.has_native_bundle() {
            self.native_rate_hz
        } else {
            self.conditioning_rate_hz
        }
    }

    pub fn engine_layout_for(&self) -> Option<EngineInputLayout> {
        if self.has_native_bundle() {
            self.native_input_layout
        } else {
            self.engine_input_layout
        }
    }

    pub fn engine_inputs(&self) -> &[PreparedChannelInput] {
        if self.has_native_bundle() {
            &self.native_inputs
        } else {
            &self.conditioning_inputs
        }
    }

    /// Display name for progress messages: "#1 Snare Drum" or "#1" if unnamed.
    pub fn display_name(&self) -> String {
        let num = self.index + 1;
        if self.name.trim().is_empty() {
            format!("#{num}")
        } else {
            format!("#{num} {}", self.name.trim())
        }
    }
}

/// Directory for the persistent upscaled sample cache.
pub fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("quinlight-audio/sample-cache")
}

/// Structured cache filename: `Quinlight-{pcm_hex}-{engine_id}-ddim{steps}.flac`
fn cache_flac_path(
    dir: &Path,
    pcm_sha256: &[u8; 32],
    engine_cache_id: &str,
    ddim_steps: u16,
) -> PathBuf {
    let pcm = pcm_hex_prefix(pcm_sha256);
    dir.join(format!(
        "Quinlight-{pcm}-{engine_cache_id}-ddim{ddim_steps}.flac"
    ))
}

/// Old opaque-hash filename for backward-compatible reads.
fn legacy_cache_wav_zst_path(dir: &Path, engine_cache_key: &str) -> PathBuf {
    dir.join(format!("Quinlight-{engine_cache_key}.wav.zst"))
}

const WAV_FORMAT_PCM: u16 = 1;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;
const WAV_FORMAT_EXTENSIBLE: u16 = 0xfffe;
const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];
const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavHeaderInfo {
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    sample_format: hound::SampleFormat,
    data_offset: usize,
    data_len: usize,
}

fn parse_wav_header(wav: &[u8]) -> Option<WavHeaderInfo> {
    if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }

    let mut pos = 12;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut sample_format = None;
    let mut data_offset = None;
    let mut data_len = 0usize;

    while pos + 8 <= wav.len() {
        let chunk_id = &wav[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([wav[pos + 4], wav[pos + 5], wav[pos + 6], wav[pos + 7]]) as usize;
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start.saturating_add(chunk_size).min(wav.len());
        if chunk_data_end < chunk_data_start {
            return None;
        }

        if chunk_id == b"fmt " {
            if chunk_size < 16 || chunk_data_end < chunk_data_start + 16 {
                return None;
            }

            let format_tag = u16::from_le_bytes([wav[chunk_data_start], wav[chunk_data_start + 1]]);
            channels = u16::from_le_bytes([wav[chunk_data_start + 2], wav[chunk_data_start + 3]]);
            sample_rate = u32::from_le_bytes([
                wav[chunk_data_start + 4],
                wav[chunk_data_start + 5],
                wav[chunk_data_start + 6],
                wav[chunk_data_start + 7],
            ]);
            bits_per_sample =
                u16::from_le_bytes([wav[chunk_data_start + 14], wav[chunk_data_start + 15]]);

            sample_format = Some(match format_tag {
                WAV_FORMAT_PCM => hound::SampleFormat::Int,
                WAV_FORMAT_IEEE_FLOAT => hound::SampleFormat::Float,
                WAV_FORMAT_EXTENSIBLE => {
                    if chunk_size < 40 || chunk_data_end < chunk_data_start + 40 {
                        return None;
                    }
                    let valid_bits_per_sample = u16::from_le_bytes([
                        wav[chunk_data_start + 18],
                        wav[chunk_data_start + 19],
                    ]);
                    let mut subformat = [0u8; 16];
                    subformat.copy_from_slice(&wav[chunk_data_start + 24..chunk_data_start + 40]);
                    if valid_bits_per_sample > 0 {
                        bits_per_sample = valid_bits_per_sample;
                    }
                    match subformat {
                        KSDATAFORMAT_SUBTYPE_PCM => hound::SampleFormat::Int,
                        KSDATAFORMAT_SUBTYPE_IEEE_FLOAT => hound::SampleFormat::Float,
                        _ => return None,
                    }
                }
                _ => return None,
            });
        } else if chunk_id == b"data" {
            data_offset = Some(chunk_data_start);
            data_len = chunk_data_end - chunk_data_start;
        }

        pos = chunk_data_start + ((chunk_size + 1) & !1);
    }

    Some(WavHeaderInfo {
        channels,
        sample_rate,
        bits_per_sample,
        sample_format: sample_format?,
        data_offset: data_offset?,
        data_len,
    })
    .filter(|header| header.channels > 0 && header.sample_rate > 0 && header.bits_per_sample > 0)
}

/// Build a 64-bit float WAV file in memory (RIFF/WAVE, IEEE float, 64 bps).
fn build_wav_f64(data: &[f64], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_size = (data.len() * 8) as u32;
    let file_size = 36 + data_size; // RIFF header excludes first 8 bytes
    let byte_rate = sample_rate * channels as u32 * 8;
    let block_align = channels * 8;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&64u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &sample in data {
        buf.extend_from_slice(&sample.to_le_bytes());
    }
    buf
}

/// Parse a 64-bit float WAV from raw bytes. Returns (samples, channels, sample_rate).
fn parse_wav_f64(wav: &[u8]) -> Option<(Vec<f64>, u16, u32)> {
    let header = parse_wav_header(wav)?;
    if header.sample_format != hound::SampleFormat::Float
        || header.bits_per_sample != 64
        || header.data_len < 8
        || header.data_len % 8 != 0
    {
        return None;
    }

    let data_bytes = &wav[header.data_offset..header.data_offset + header.data_len];
    let samples: Vec<f64> = data_bytes
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    Some((samples, header.channels, header.sample_rate))
}

/// Look up a cached upscaled sample.
/// Tries the FLAC format first, then legacy zstd WAV formats.
/// Returns `None` immediately if `--no-cache` is active.
fn cache_lookup(
    pcm_sha256: &[u8; 32],
    engine_cache_id: &str,
    engine_cache_key: &str,
    ddim_steps: u16,
    engine_name: &str,
    index: i32,
    sample_rate_hz: i32,
) -> Option<SampleResult> {
    if NO_CACHE.load(Ordering::Relaxed) {
        return None;
    }
    let dir = cache_dir();

    // Current: Quinlight-{pcm_hex}-{engine_id}-ddim{steps}.flac
    let flac = cache_flac_path(&dir, pcm_sha256, engine_cache_id, ddim_steps);
    if let Some(result) = read_flac_cache(&flac, engine_name, index, sample_rate_hz) {
        return Some(result);
    }

    // Legacy: Quinlight-{pcm_hex}-{engine_id}-ddim{steps}.wav.zst
    let wav_zst = dir.join(format!(
        "Quinlight-{}-{engine_cache_id}-ddim{ddim_steps}.wav.zst",
        pcm_hex_prefix(pcm_sha256)
    ));
    if let Some(result) = read_wav_zst_cache(&wav_zst, engine_name, index, sample_rate_hz) {
        return Some(result);
    }

    // Very old: Quinlight-{opaque_hash}.wav.zst
    let legacy = legacy_cache_wav_zst_path(&dir, engine_cache_key);
    if let Some(result) = read_wav_zst_cache(&legacy, engine_name, index, sample_rate_hz) {
        return Some(result);
    }

    None
}

fn read_flac_cache(
    path: &Path,
    engine_name: &str,
    index: i32,
    sample_rate_hz: i32,
) -> Option<SampleResult> {
    use flac_codec::decode::{FlacSampleReader, Metadata};
    let bytes = std::fs::read(path).ok()?;
    let reader = match FlacSampleReader::new(std::io::Cursor::new(bytes)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Cache: corrupt FLAC file {}: {e:?}", path.display());
            return None;
        }
    };
    let channels = reader.channel_count() as i32;
    let i32_samples: Vec<i32> = reader.into_iter().collect::<Result<_, _>>().ok()?;
    let data: Vec<f64> = i32_samples
        .iter()
        .map(|&s| s as f64 / FLAC_I32_SCALE)
        .collect();
    let frames = if channels > 0 {
        data.len() as i64 / channels as i64
    } else {
        0
    };
    Some(SampleResult {
        index,
        data,
        length_frames: frames,
        channels,
        sample_rate_hz,
        engine_name: engine_name.to_string(),
        discovered_loops: None,
    })
}

fn read_wav_zst_cache(
    path: &Path,
    engine_name: &str,
    index: i32,
    sample_rate_hz: i32,
) -> Option<SampleResult> {
    let compressed = std::fs::read(path).ok()?;
    let wav = match zstd::decode_all(compressed.as_slice()) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Cache: corrupt zstd file {}: {e}", path.display());
            return None;
        }
    };
    let (data, channels, wav_rate) = parse_wav_f64(&wav)?;
    debug_assert_eq!(
        wav_rate, sample_rate_hz as u32,
        "WAV header rate {wav_rate} != expected {sample_rate_hz}"
    );
    let ch = channels as i32;
    let frames = if ch > 0 {
        data.len() as i64 / ch as i64
    } else {
        0
    };
    Some(SampleResult {
        index,
        data,
        length_frames: frames,
        channels: ch,
        sample_rate_hz,
        engine_name: engine_name.to_string(),
        discovered_loops: None,
    })
}

/// Store an upscaled sample in the cache as normalized 32-bit integer FLAC.
fn cache_store(
    pcm_sha256: &[u8; 32],
    engine_cache_id: &str,
    ddim_steps: u16,
    result: &SampleResult,
    target_rms: f64,
) {
    if NO_CACHE.load(Ordering::Relaxed) {
        return;
    }
    let dir = cache_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    // Normalize a copy so values are in [-1.0, 1.0] for int32 quantization
    let mut normalized = result.clone();
    normalize_sample(&mut normalized, target_rms);

    let pcm = pcm_hex_prefix(pcm_sha256);
    let tmp_path = dir.join(format!(
        "Quinlight-{pcm}-{engine_cache_id}-ddim{ddim_steps}.{:?}.flac.tmp",
        std::thread::current().id()
    ));
    let final_path = cache_flac_path(&dir, pcm_sha256, engine_cache_id, ddim_steps);

    let flac = match build_flac_i32(
        &normalized.data,
        normalized.sample_rate_hz.max(1) as u32,
        normalized.channels.max(1) as u16,
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cache: FLAC encode failed: {e}");
            return;
        }
    };

    if std::fs::write(&tmp_path, &flac).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }

    // Atomic rename
    let _ = std::fs::rename(&tmp_path, &final_path);
}

/// Scale factor for f64 ↔ i32 FLAC conversion.
const FLAC_I32_SCALE: f64 = i32::MAX as f64;

fn build_flac_i32(data: &[f64], sample_rate: u32, channels: u16) -> Result<Vec<u8>, String> {
    use flac_codec::encode::{FlacSampleWriter, Options};
    let mut buf = std::io::Cursor::new(Vec::new());
    let total = data.len() as u64;
    let mut writer = FlacSampleWriter::new(
        &mut buf,
        Options::default(),
        sample_rate,
        32,
        channels as u8,
        Some(total),
    )
    .map_err(|e| format!("FLAC encoder init: {e:?}"))?;
    let i32_samples: Vec<i32> = data
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * FLAC_I32_SCALE) as i32)
        .collect();
    writer
        .write(&i32_samples)
        .map_err(|e| format!("FLAC encode: {e:?}"))?;
    writer
        .finalize()
        .map_err(|e| format!("FLAC finalize: {e:?}"))?;
    Ok(buf.into_inner())
}

/// Remove DC bias (per-channel) and match RMS to the canonical reference.
fn normalize_sample(result: &mut SampleResult, target_rms: f64) {
    let channels = result.channels as usize;
    if channels == 0 || result.data.is_empty() {
        return;
    }

    remove_dc_per_channel(&mut result.data, channels);

    let upscaled_rms = simd::rms_f64(&result.data);
    if upscaled_rms > 1e-10 && target_rms > 1e-10 {
        let gain = target_rms / upscaled_rms;
        simd::scale_in_place_f64(&mut result.data, gain);
    }
}

const SAMPLE_CACHE_HASH_VERSION: u8 = 13;
const PREVIOUS_SAMPLE_CACHE_HASH_VERSION: u8 = 12;
const LEGACY_SAMPLE_CACHE_HASH_VERSION: u8 = 2;

/// Bumped whenever the AI-facing conditioning buffer or the extraction
/// pipeline that reads the engine output changes, so caches baked by an
/// earlier pipeline don't silently satisfy lookups under the new one.
/// Bump this constant in the same commit that changes `pad_for_engine`,
/// `extract_channel_result`, or the tiling layouts they produce.
const ENGINE_PIPELINE_VERSION: u8 = 4;

#[derive(Debug, Clone, Copy)]
struct SampleHashLoopMeta {
    looped: bool,
    normal_loop: SampleLoopRegion,
    sustain_loop: SampleLoopRegion,
    saved_length_frames: i64,
    source_length_frames: i64,
    prep_mode: LoopPrepMode,
}

impl SampleHashLoopMeta {
    fn legacy(looped: bool, data_len: usize, channels: i32) -> Self {
        let frames = if channels > 0 {
            data_len as i64 / channels as i64
        } else {
            0
        };
        Self {
            looped,
            normal_loop: if looped {
                SampleLoopRegion::forward(0, frames)
            } else {
                SampleLoopRegion::none()
            },
            sustain_loop: SampleLoopRegion::none(),
            saved_length_frames: frames,
            source_length_frames: frames,
            prep_mode: if looped {
                LoopPrepMode::WholeSampleLoopFallback
            } else {
                LoopPrepMode::OneShot
            },
        }
    }

    fn from_loop_prep(loop_prep: LoopPrepPlan) -> Self {
        Self {
            looped: loop_prep.is_looped(),
            normal_loop: loop_prep.normal_loop,
            sustain_loop: loop_prep.sustain_loop,
            saved_length_frames: loop_prep.saved_length_frames,
            source_length_frames: loop_prep.source_length_frames,
            prep_mode: loop_prep.mode,
        }
    }
}

fn compute_sample_hash_with_loop_meta(
    data: &[f64],
    rate: i32,
    channels: i32,
    loop_meta: SampleHashLoopMeta,
    cleanup_tag: u8,
    version: u8,
) -> String {
    let mut hasher = Sha256::new();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    hasher.update(bytes);
    hasher.update(rate.to_le_bytes());
    hasher.update(channels.to_le_bytes());
    hasher.update([loop_meta.looped as u8]);
    hasher.update(loop_meta.normal_loop.start_frames.to_le_bytes());
    hasher.update(loop_meta.normal_loop.end_frames.to_le_bytes());
    hasher.update([loop_meta.normal_loop.mode as u8]);
    hasher.update(loop_meta.sustain_loop.start_frames.to_le_bytes());
    hasher.update(loop_meta.sustain_loop.end_frames.to_le_bytes());
    hasher.update([loop_meta.sustain_loop.mode as u8]);
    hasher.update(loop_meta.saved_length_frames.to_le_bytes());
    hasher.update(loop_meta.source_length_frames.to_le_bytes());
    hasher.update([match loop_meta.prep_mode {
        LoopPrepMode::OneShot => 0,
        LoopPrepMode::WholeSampleLoopFallback => 1,
        LoopPrepMode::BoundaryLoop => 2,
        LoopPrepMode::SustainPacked => 3,
    }]);
    hasher.update([cleanup_tag]);
    hasher.update([version]);
    format!("{:x}", hasher.finalize())
}

fn compute_sample_hash_with_version(
    data: &[f64],
    rate: i32,
    channels: i32,
    looped: bool,
    cleanup_tag: u8,
    version: u8,
) -> String {
    compute_sample_hash_with_loop_meta(
        data,
        rate,
        channels,
        SampleHashLoopMeta::legacy(looped, data.len(), channels),
        cleanup_tag,
        version,
    )
}

fn compute_sample_hash_for_loop_prep(
    data: &[f64],
    rate: i32,
    channels: i32,
    loop_prep: LoopPrepPlan,
    cleanup_settings: CleanupSettings,
) -> String {
    compute_sample_hash_with_loop_meta(
        data,
        rate,
        channels,
        SampleHashLoopMeta::from_loop_prep(loop_prep),
        cleanup_settings.hash_tag(),
        SAMPLE_CACHE_HASH_VERSION,
    )
}

/// Compute SHA-256 hash of the canonical reference sample copy + rate + channels + preset.
#[allow(dead_code)]
fn compute_sample_hash(
    data: &[f64],
    rate: i32,
    channels: i32,
    looped: bool,
    cleanup_settings: CleanupSettings,
) -> String {
    compute_sample_hash_with_version(
        data,
        rate,
        channels,
        looped,
        cleanup_settings.hash_tag(),
        SAMPLE_CACHE_HASH_VERSION,
    )
}

fn compute_legacy_sample_hash(data: &[f64], rate: i32, channels: i32, looped: bool) -> String {
    let mut hasher = Sha256::new();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    hasher.update(bytes);
    hasher.update(rate.to_le_bytes());
    hasher.update(channels.to_le_bytes());
    hasher.update([looped as u8]);
    hasher.update([LEGACY_SAMPLE_CACHE_HASH_VERSION]);
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Spec-compliant cache key (two-level SHA-256)
// ---------------------------------------------------------------------------

/// Inner hash: SHA-256 of the prepared PCM bytes, independent of engine.
pub(crate) fn compute_pcm_sha256(data: &[f64]) -> [u8; 32] {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    Sha256::digest(bytes).into()
}

/// First 16 hex chars of `pcm_sha256`, used as a filename prefix for cache clearing.
fn pcm_hex_prefix(pcm_sha256: &[u8; 32]) -> String {
    pcm_sha256[..8]
        .iter()
        .fold(String::with_capacity(16), |mut s, b| {
            use std::fmt::Write;
            write!(s, "{b:02x}").ok();
            s
        })
}

/// Map loop metadata to the spec's (loop_start, loop_end, loop_type) triple.
fn spec_loop_fields(loop_info: &SampleLoopInfo) -> (i32, i32, u8) {
    if loop_info.normal.has_loop() {
        let loop_type = match loop_info.normal.mode {
            SampleLoopMode::None => 0u8,
            SampleLoopMode::Forward => 1u8,
            SampleLoopMode::PingPong => 2u8,
        };
        (
            loop_info.normal.start_frames as i32,
            loop_info.normal.end_frames as i32,
            loop_type,
        )
    } else {
        (-1i32, -1i32, 0u8)
    }
}

/// Per-engine cache key per the spec's exact binary layout:
/// `pcm_sha256 || rate || depth || loop_start || loop_end || loop_type
///  || target_rate || engine_id\0 || ddim_steps || pipeline_version`
fn compute_engine_cache_key(
    pcm_sha256: &[u8; 32],
    source_sample_rate: u32,
    source_bit_depth: u8,
    loop_start: i32,
    loop_end: i32,
    loop_type: u8,
    target_sample_rate: u32,
    engine_cache_id: &str,
    ddim_steps: u16,
    pipeline_version: u8,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(pcm_sha256);
    hasher.update(source_sample_rate.to_le_bytes());
    hasher.update([source_bit_depth]);
    hasher.update(loop_start.to_le_bytes());
    hasher.update(loop_end.to_le_bytes());
    hasher.update([loop_type]);
    hasher.update(target_sample_rate.to_le_bytes());
    hasher.update(engine_cache_id.as_bytes());
    hasher.update([0u8]); // null terminator
    hasher.update(ddim_steps.to_le_bytes());
    hasher.update([pipeline_version]);
    format!("{:x}", hasher.finalize())
}

/// Convenience: compute engine cache key from a SampleJob and engine metadata.
fn engine_cache_key_for_job(job: &SampleJob, engine_cache_id: &str, ddim_steps: u16) -> String {
    let (loop_start, loop_end, loop_type) = spec_loop_fields(&job.loop_info);
    compute_engine_cache_key(
        &job.pcm_sha256,
        job.rate.max(0) as u32,
        job.bits_per_sample,
        loop_start,
        loop_end,
        loop_type,
        job.output_sample_rate_hz.max(0) as u32,
        engine_cache_id,
        ddim_steps,
        ENGINE_PIPELINE_VERSION,
    )
}

/// Collect all cache identifiers for a single sample across every cleanup variant.
/// Returns pcm_sha256 hex prefixes (match structured filenames via directory scan)
/// and legacy hashes (match old opaque-hash filenames via exact/prefix lookup).
fn collect_cache_hashes_for_sample(
    original_data: &[f64],
    rate: i32,
    channels: i32,
    loop_prep: LoopPrepPlan,
) -> Vec<String> {
    let mut hashes = Vec::new();

    if let Ok(current_reference_48k) =
        build_current_pipeline_reference_48k(original_data, rate as u32, channels as usize)
    {
        hashes.push(pcm_hex_prefix(&compute_pcm_sha256(&current_reference_48k)));
    }

    let off_reference = build_canonical_reference_with_loop_prep(
        original_data,
        rate as u32,
        channels as usize,
        loop_prep,
        CleanupSettings::off(),
    )
    .unwrap_or_default();

    // pcm_sha256 prefix — matches structured filenames like Quinlight-{prefix}-*.wav.zst
    hashes.push(pcm_hex_prefix(&compute_pcm_sha256(&off_reference)));

    // Legacy hashes (old format, for deletion of old cache files)
    hashes.push(compute_legacy_sample_hash(
        &off_reference,
        rate,
        channels,
        loop_prep.is_looped(),
    ));
    hashes.push(compute_sample_hash_with_version(
        &off_reference,
        rate,
        channels,
        loop_prep.is_looped(),
        CleanupSettings::off().hash_tag(),
        PREVIOUS_SAMPLE_CACHE_HASH_VERSION,
    ));
    hashes.push(compute_sample_hash_for_loop_prep(
        &off_reference,
        rate,
        channels,
        loop_prep,
        CleanupSettings::off(),
    ));

    for cleanup_settings in CleanupSettings::ALL_ACTIVE {
        if let Ok(reference) = build_canonical_reference_with_loop_prep(
            original_data,
            rate as u32,
            channels as usize,
            loop_prep,
            cleanup_settings,
        ) {
            hashes.push(pcm_hex_prefix(&compute_pcm_sha256(&reference)));

            // Legacy hashes
            hashes.push(compute_sample_hash_for_loop_prep(
                &reference,
                rate,
                channels,
                loop_prep,
                cleanup_settings,
            ));
            hashes.push(compute_sample_hash_with_version(
                &reference,
                rate,
                channels,
                loop_prep.is_looped(),
                cleanup_settings.hash_tag(),
                PREVIOUS_SAMPLE_CACHE_HASH_VERSION,
            ));
        }
    }

    for cleanup_preset in [RetiredCleanupPreset::Light, RetiredCleanupPreset::Archival] {
        if let Ok(reference) = build_retired_cleanup_reference_with_loop_prep(
            original_data,
            rate as u32,
            channels as usize,
            loop_prep,
            cleanup_preset,
        ) {
            hashes.push(pcm_hex_prefix(&compute_pcm_sha256(&reference)));
            hashes.push(compute_sample_hash_with_version(
                &reference,
                rate,
                channels,
                loop_prep.is_looped(),
                cleanup_preset.hash_tag(),
                PREVIOUS_SAMPLE_CACHE_HASH_VERSION,
            ));
        }
    }

    hashes
}

/// Collect all cache hashes that would exist for a module's samples across
/// every cleanup variant (current, previous-version, retired, and legacy).
pub fn collect_cache_hashes_for_module(module: &mut Module) -> Vec<String> {
    let num_samples = module.num_samples();
    let mut all_hashes = Vec::new();

    for i in 0..num_samples {
        let rate = module.sample_rate(i);
        let length = module.sample_length_frames(i);
        let channels = module.sample_channels(i);
        if length <= 0 || channels <= 0 || rate >= 48000 {
            continue;
        }
        let Some(original_data) = module.read_sample_data(i) else {
            continue;
        };
        let loop_info = module.sample_loop_info(i);
        let loop_prep = LoopPrepPlan::from_sample(length, loop_info);
        all_hashes.extend(collect_cache_hashes_for_sample(
            &original_data,
            rate,
            channels,
            loop_prep,
        ));
    }

    all_hashes.sort();
    all_hashes.dedup();
    all_hashes
}

/// Delete cache files matching the given hashes from the cache directory.
pub fn delete_cache_files_for_hashes(hashes: &[String]) -> usize {
    let cache = cache_dir();
    let mut deleted = 0;

    // Direct-path deletions for new spec format and legacy formats
    for hash in hashes {
        // New spec format: Quinlight-{engine_cache_key}.wav.zst
        let new_path = cache.join(format!("Quinlight-{hash}.wav.zst"));
        if std::fs::remove_file(&new_path).is_ok() {
            deleted += 1;
        }
        // Very old single-file legacy
        let legacy_path = cache.join(format!("{hash}.flac"));
        if std::fs::remove_file(&legacy_path).is_ok() {
            deleted += 1;
        }
    }

    // Directory scan for legacy prefix-matched files (engine name in filename)
    let old_prefixes: Vec<String> = hashes.iter().map(|h| format!("{h}-")).collect();
    let old_new_prefixes: Vec<String> = hashes.iter().map(|h| format!("Quinlight-{h}-")).collect();
    if let Ok(entries) = std::fs::read_dir(&cache) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.contains(".tmp") {
                continue;
            }
            let prefixes_match = old_new_prefixes
                .iter()
                .any(|p| name.starts_with(p.as_str()));
            let is_match = (name.ends_with(".flac") || name.ends_with(".wav.zst"))
                && prefixes_match
                || (name.ends_with(".flac")
                    && old_prefixes.iter().any(|p| name.starts_with(p.as_str())));
            if is_match {
                if std::fs::remove_file(entry.path()).is_ok() {
                    deleted += 1;
                }
            }
        }
    }

    deleted
}

/// Delete cached upscaled samples for every sample in the currently loaded module.
/// Returns the number of cache entries deleted.
pub fn clear_cache_for_module(module: &mut Module) -> usize {
    let hashes = collect_cache_hashes_for_module(module);
    delete_cache_files_for_hashes(&hashes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleBoundaryMode {
    OneShot,
    #[allow(dead_code)] // Constructed only in tests; matched in production resample_audio()
    LoopAware,
}

pub(crate) fn scaled_frame_count(input_frames: usize, input_rate: u32, output_rate: u32) -> usize {
    if input_frames == 0 || input_rate == 0 || output_rate == 0 {
        return 0;
    }

    let scaled = ((input_frames as u128 * output_rate as u128) + (input_rate as u128 / 2))
        / input_rate as u128;
    scaled.max(1).min(usize::MAX as u128) as usize
}

pub(crate) fn linear_resample_interleaved(
    data: &[f64],
    input_rate: u32,
    output_rate: u32,
    channels: usize,
) -> Vec<f64> {
    if channels == 0 || data.is_empty() || input_rate == 0 || output_rate == 0 {
        return Vec::new();
    }

    let input_frames = data.len() / channels;
    if input_frames == 0 {
        return Vec::new();
    }

    let output_frames = scaled_frame_count(input_frames, input_rate, output_rate);
    let mut output = vec![0.0f64; output_frames * channels];
    for frame in 0..output_frames {
        let src = frame as f64 * input_rate as f64 / output_rate as f64;
        let src_idx = src.floor() as usize;
        let next_idx = (src_idx + 1).min(input_frames - 1);
        let frac = (src - src_idx as f64) as f64;
        for ch in 0..channels {
            let a = data[src_idx * channels + ch];
            let b = data[next_idx * channels + ch];
            output[frame * channels + ch] = a * (1.0 - frac) + b * frac;
        }
    }
    output
}

fn ffmpeg_channel_layout(channels: usize) -> Option<ChannelLayout> {
    match channels {
        1 => Some(get_channel_layout("mono")),
        2 => Some(get_channel_layout("stereo")),
        _ => None,
    }
}

pub(crate) fn ffmpeg_input_sample_format() -> FfmpegSampleFormat {
    get_sample_format("dbl")
}

pub(crate) fn copy_interleaved_f64_to_audio_frame(
    frame: &mut AudioFrameMut,
    chunk: &[f64],
) -> Result<(), String> {
    let mut planes = frame.planes_mut();
    let plane = planes
        .first_mut()
        .ok_or_else(|| "FFmpeg returned no audio planes".to_string())?;
    let data = plane.data_mut();
    let byte_len = std::mem::size_of_val(chunk);
    if data.len() < byte_len {
        return Err("FFmpeg audio frame plane is smaller than the source chunk".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(chunk.as_ptr().cast::<u8>(), data.as_mut_ptr(), byte_len);
    }
    Ok(())
}

fn extend_interleaved_f64_from_audio_frame(
    frame: &AudioFrame,
    channels: usize,
    output: &mut Vec<f64>,
) -> Result<(), String> {
    let planes = frame.planes();
    let plane = planes
        .first()
        .ok_or_else(|| "FFmpeg resampler returned no audio planes".to_string())?;
    let sample_count = frame.samples() * channels;
    let byte_len = sample_count * std::mem::size_of::<f64>();
    let data = plane.data();
    if data.len() < byte_len {
        return Err("FFmpeg resampler frame was smaller than expected".into());
    }
    let base = output.len();
    output.reserve(sample_count);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            output.as_mut_ptr().add(base).cast::<u8>(),
            byte_len,
        );
        output.set_len(base + sample_count);
    }
    Ok(())
}

fn drain_resampler_output(
    resampler: &mut AudioResampler,
    channels: usize,
    output: &mut Vec<f64>,
) -> Result<(), String> {
    while let Some(frame) = resampler
        .take()
        .map_err(|e| format!("FFmpeg resampler frame error: {e}"))?
    {
        extend_interleaved_f64_from_audio_frame(&frame, channels, output)?;
    }
    Ok(())
}

/// Resample audio data from one sample rate to another using FFmpeg's audio resampler.
/// Input and output are interleaved float audio.
fn resample_audio_one_shot(
    data: &[f64],
    input_rate: u32,
    output_rate: u32,
    channels: usize,
) -> Result<Vec<f64>, String> {
    if channels == 0 || data.is_empty() {
        return Ok(vec![]);
    }
    if input_rate == 0 || output_rate == 0 {
        return Ok(vec![]);
    }
    if input_rate == output_rate {
        return Ok(data.to_vec());
    }
    if channels > 2 {
        return Ok(linear_resample_interleaved(
            data,
            input_rate,
            output_rate,
            channels,
        ));
    }

    let frames = data.len() / channels;
    if frames == 0 {
        return Ok(Vec::new());
    }

    let channel_layout = ffmpeg_channel_layout(channels)
        .ok_or_else(|| format!("Unsupported resample channel count: {channels}"))?;
    let sample_format = ffmpeg_input_sample_format();
    let sample_time_base = TimeBase::new(1, input_rate as i32);
    let chunk_frames = frames.min(4096).max(1);
    let mut next_pts = 0i64;
    let mut resampler = AudioResampler::builder()
        .source_channel_layout(channel_layout.clone())
        .source_sample_format(sample_format)
        .source_sample_rate(input_rate)
        .target_channel_layout(channel_layout.clone())
        .target_sample_format(sample_format)
        .target_sample_rate(output_rate)
        .target_frame_samples(Some(chunk_frames))
        .build()
        .map_err(|e| format!("Failed to create FFmpeg audio resampler: {e}"))?;
    let mut output =
        Vec::with_capacity(scaled_frame_count(frames, input_rate, output_rate) * channels);

    for chunk in data.chunks(chunk_frames * channels) {
        let frame_samples = chunk.len() / channels;
        if frame_samples == 0 {
            continue;
        }
        let mut frame =
            AudioFrameMut::silence(&channel_layout, sample_format, input_rate, frame_samples)
                .with_time_base(sample_time_base)
                .with_pts(Timestamp::new(next_pts, sample_time_base));
        next_pts += frame_samples as i64;
        copy_interleaved_f64_to_audio_frame(&mut frame, chunk)?;
        resampler
            .push(frame.freeze())
            .map_err(|e| format!("FFmpeg resampler push error: {e}"))?;
        drain_resampler_output(&mut resampler, channels, &mut output)?;
    }

    resampler
        .flush()
        .map_err(|e| format!("FFmpeg resampler flush error: {e}"))?;
    drain_resampler_output(&mut resampler, channels, &mut output)?;

    let expected_samples = scaled_frame_count(frames, input_rate, output_rate) * channels;
    if output.len() > expected_samples {
        output.truncate(expected_samples);
    } else if output.len() < expected_samples {
        if output.is_empty() {
            output = linear_resample_interleaved(data, input_rate, output_rate, channels);
        } else {
            let last_frame_start = output.len().saturating_sub(channels);
            let last_frame: Vec<f64> = output[last_frame_start..].to_vec();
            while output.len() < expected_samples {
                output.extend_from_slice(&last_frame);
            }
            output.truncate(expected_samples);
        }
    }
    Ok(output)
}

/// Resample sample data from one sample rate to another, using either one-shot or
/// circular loop-aware boundary handling.
pub fn resample_audio(
    data: &[f64],
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    boundary_mode: ResampleBoundaryMode,
) -> Result<Vec<f64>, String> {
    match boundary_mode {
        ResampleBoundaryMode::OneShot => {
            resample_audio_one_shot(data, input_rate, output_rate, channels)
        }
        ResampleBoundaryMode::LoopAware => {
            if channels == 0 || data.is_empty() {
                return Ok(vec![]);
            }

            let frames = data.len() / channels;
            let one_copy_frames = scaled_frame_count(frames, input_rate, output_rate);
            let one_copy_samples = one_copy_frames * channels;

            let mut tiled = Vec::with_capacity(data.len() * 3);
            tiled.extend_from_slice(data);
            tiled.extend_from_slice(data);
            tiled.extend_from_slice(data);

            let resampled = resample_audio_one_shot(&tiled, input_rate, output_rate, channels)?;
            let start = one_copy_samples;
            let end = start + one_copy_samples;
            if resampled.len() < end {
                return Err(format!(
                    "Loop-aware resample produced too little output: need {end} samples, got {}",
                    resampled.len()
                ));
            }

            Ok(resampled[start..end].to_vec())
        }
    }
}

/// Shared prepared basis for the current simplified remaster pipeline:
/// one-shot resample to 48 kHz, with per-channel DC bias removed so the
/// resampled buffer matches the AI-path invariant (zero-mean per channel).
pub(crate) fn build_current_pipeline_reference_48k(
    original_data: &[f64],
    original_rate: u32,
    channels: usize,
) -> Result<Vec<f64>, String> {
    if channels == 0 || original_data.is_empty() || original_rate == 0 {
        return Ok(Vec::new());
    }

    let mut resampled = resample_audio_one_shot(original_data, original_rate, 48_000, channels)?;
    remove_dc_per_channel(&mut resampled, channels);
    Ok(resampled)
}

// libopenmpt command indices (must match OPENMPT_MODULE_COMMAND_* constants)
const COMMAND_NOTE: i32 = 0;
const COMMAND_INSTRUMENT: i32 = 1;
const COMMAND_EFFECT: i32 = 3;
const COMMAND_PARAMETER: i32 = 5;

// Effect command types (from modcommand.h)
const CMD_OFFSET: u8 = 10; // Oxx: sample offset
const CMD_S3MCMDEX: u8 = 20; // Sxx: S3M extended commands (SAx = high offset)

fn module_uses_xm_offsets(module: &Module) -> bool {
    module.info().format_type.eq_ignore_ascii_case("xm")
}

fn patch_sample_offsets_non_xm(
    module: &mut Module,
    sample_index: i32,
    old_rate: i32,
    new_rate: i32,
) {
    let instrument = (sample_index + 1) as u8; // S3M/IT: instrument = sample + 1
    let num_patterns = module.num_patterns();
    let num_channels = module.num_channels();
    let old_r = old_rate as u32;
    let new_r = new_rate as u32;

    for pat in 0..num_patterns {
        let num_rows = module.pattern_num_rows(pat);
        if num_rows <= 0 {
            continue;
        }
        let mut channel_instrument = vec![0u8; num_channels as usize];

        for row in 0..num_rows {
            for ch in 0..num_channels {
                let instr = module.get_pattern_command(pat, row, ch, COMMAND_INSTRUMENT);
                if instr > 0 {
                    channel_instrument[ch as usize] = instr;
                }

                if channel_instrument[ch as usize] != instrument {
                    continue;
                }

                let effect = module.get_pattern_command(pat, row, ch, COMMAND_EFFECT);
                let param = module.get_pattern_command(pat, row, ch, COMMAND_PARAMETER);

                let new_param = match effect {
                    CMD_OFFSET if param > 0 => {
                        Some(((param as u32) * new_r / old_r).min(255) as u8)
                    }
                    CMD_S3MCMDEX if (param & 0xF0) == 0xA0 => {
                        let old_val = (param & 0x0F) as u32;
                        let new_val = ((old_val << 16) * new_r / old_r) >> 16;
                        Some(0xA0 | new_val.min(15) as u8)
                    }
                    _ => None,
                };

                if let Some(p) = new_param {
                    module.set_pattern_command(pat, row, ch, COMMAND_PARAMETER, p);
                }
            }
        }
    }
}

fn patch_sample_offsets_xm(module: &mut Module, sample_index: i32, old_rate: i32, new_rate: i32) {
    let num_patterns = module.num_patterns();
    let num_channels = module.num_channels();
    let old_r = old_rate as u32;
    let new_r = new_rate as u32;

    for pat in 0..num_patterns {
        let num_rows = module.pattern_num_rows(pat);
        if num_rows <= 0 {
            continue;
        }
        let mut channel_instrument = vec![0u8; num_channels as usize];

        for row in 0..num_rows {
            for ch in 0..num_channels {
                let instr = module.get_pattern_command(pat, row, ch, COMMAND_INSTRUMENT);
                if instr > 0 {
                    channel_instrument[ch as usize] = instr;
                }

                let note = module.get_pattern_command(pat, row, ch, COMMAND_NOTE);
                let instrument = channel_instrument[ch as usize];
                if note == 0 || instrument == 0 {
                    continue;
                }

                let Some(mapped_sample) =
                    module.instrument_sample_for_note((instrument - 1) as i32, note)
                else {
                    continue;
                };
                if mapped_sample != sample_index {
                    continue;
                }

                let effect = module.get_pattern_command(pat, row, ch, COMMAND_EFFECT);
                let param = module.get_pattern_command(pat, row, ch, COMMAND_PARAMETER);
                if effect == CMD_OFFSET && param > 0 {
                    let new_param = ((param as u32) * new_r / old_r).min(255) as u8;
                    module.set_pattern_command(pat, row, ch, COMMAND_PARAMETER, new_param);
                }
            }
        }
    }
}

/// Patch sample offset effects in all patterns after remastering a sample.
/// Scales Oxx/SAx by new_rate/old_rate (frame positions scale up with sample rate).
/// Other rate-sensitive pitch effects are compensated in the C++ playback engine.
/// `sample_index` is 0-based.
pub fn patch_sample_offsets(module: &mut Module, sample_index: i32, old_rate: i32, new_rate: i32) {
    if old_rate <= 0 || new_rate <= 0 || old_rate == new_rate {
        return;
    }
    if module_uses_xm_offsets(module) {
        patch_sample_offsets_xm(module, sample_index, old_rate, new_rate);
    } else {
        patch_sample_offsets_non_xm(module, sample_index, old_rate, new_rate);
    }
}

/// Saved effect param: (pattern, row, channel, original_param).
pub type SavedEffectParam = (i32, i32, i32, u8);

fn save_effect_params_non_xm(module: &Module, sample_index: i32) -> Vec<SavedEffectParam> {
    let instrument = (sample_index + 1) as u8;
    let num_patterns = module.num_patterns();
    let num_channels = module.num_channels();
    let mut saved = Vec::new();

    for pat in 0..num_patterns {
        let num_rows = module.pattern_num_rows(pat);
        if num_rows <= 0 {
            continue;
        }
        let mut channel_instrument = vec![0u8; num_channels as usize];

        for row in 0..num_rows {
            for ch in 0..num_channels {
                let instr = module.get_pattern_command(pat, row, ch, COMMAND_INSTRUMENT);
                if instr > 0 {
                    channel_instrument[ch as usize] = instr;
                }
                if channel_instrument[ch as usize] != instrument {
                    continue;
                }

                let effect = module.get_pattern_command(pat, row, ch, COMMAND_EFFECT);
                let param = module.get_pattern_command(pat, row, ch, COMMAND_PARAMETER);
                let dominated = (effect == CMD_OFFSET && param > 0)
                    || (effect == CMD_S3MCMDEX && (param & 0xF0) == 0xA0);

                if dominated {
                    saved.push((pat, row, ch, param));
                }
            }
        }
    }

    saved
}

fn save_effect_params_xm(module: &Module, sample_index: i32) -> Vec<SavedEffectParam> {
    let num_patterns = module.num_patterns();
    let num_channels = module.num_channels();
    let mut saved = Vec::new();

    for pat in 0..num_patterns {
        let num_rows = module.pattern_num_rows(pat);
        if num_rows <= 0 {
            continue;
        }
        let mut channel_instrument = vec![0u8; num_channels as usize];

        for row in 0..num_rows {
            for ch in 0..num_channels {
                let instr = module.get_pattern_command(pat, row, ch, COMMAND_INSTRUMENT);
                if instr > 0 {
                    channel_instrument[ch as usize] = instr;
                }

                let note = module.get_pattern_command(pat, row, ch, COMMAND_NOTE);
                let instrument = channel_instrument[ch as usize];
                if note == 0 || instrument == 0 {
                    continue;
                }

                let Some(mapped_sample) =
                    module.instrument_sample_for_note((instrument - 1) as i32, note)
                else {
                    continue;
                };
                if mapped_sample != sample_index {
                    continue;
                }

                let effect = module.get_pattern_command(pat, row, ch, COMMAND_EFFECT);
                let param = module.get_pattern_command(pat, row, ch, COMMAND_PARAMETER);
                if effect == CMD_OFFSET && param > 0 {
                    saved.push((pat, row, ch, param));
                }
            }
        }
    }

    saved
}

/// Snapshot all pattern params that `patch_sample_offsets` would modify.
/// Call this BEFORE the first patch to preserve original values for lossless restore.
pub fn save_effect_params(module: &Module, sample_index: i32) -> Vec<SavedEffectParam> {
    if module_uses_xm_offsets(module) {
        save_effect_params_xm(module, sample_index)
    } else {
        save_effect_params_non_xm(module, sample_index)
    }
}

/// Restore original pattern params from a saved snapshot.
pub fn restore_effect_params(module: &mut Module, saved: &[SavedEffectParam]) {
    for &(pat, row, ch, param) in saved {
        module.set_pattern_command(pat, row, ch, COMMAND_PARAMETER, param);
    }
}

#[allow(dead_code)]
pub(crate) fn apply_sample_replacement(
    module: &mut Module,
    sample_index: i32,
    data: &[f64],
    length_frames: i64,
    channels: i32,
    target_rate: i32,
    original_rate: i32,
    saved_effects: &[SavedEffectParam],
) -> Result<(), String> {
    if !module.replace_sample_data(sample_index, data, length_frames, channels, target_rate) {
        return Err(format!("Failed to replace sample {}", sample_index + 1));
    }
    if !saved_effects.is_empty() {
        restore_effect_params(module, saved_effects);
    }
    if target_rate != original_rate {
        patch_sample_offsets(module, sample_index, original_rate, target_rate);
    }
    Ok(())
}

/// Original sample data preserved for toggle support.
pub struct OriginalSample {
    pub index: i32,
    pub data: Vec<f64>,
    pub rate: i32,
    pub channels: i32,
    pub bits_per_sample: i32,
    pub source_length_frames: i64,
    #[allow(dead_code)] // Kept for metadata; unused after pipeline simplification
    pub effective_length_frames: i64,
    #[allow(dead_code)] // Kept for metadata; unused after pipeline simplification
    pub loop_start_frames: i64,
    pub looped: bool,
    pub loop_info: SampleLoopInfo,
    pub name: String,
}

/// Count samples that already meet or exceed the 48 kHz target rate.
/// Used by the convert pipeline to credit mods that ship high-fi samples
/// even when AI consensus isn't reached on the remaining low-rate samples.
pub fn count_high_fidelity_samples(module: &mut Module) -> usize {
    let num_samples = module.num_samples();
    let mut count = 0;
    for i in 0..num_samples {
        let length = module.sample_length_frames(i);
        let channels = module.sample_channels(i);
        let rate = module.sample_rate(i);
        if length > 0 && channels > 0 && rate >= 48000 {
            count += 1;
        }
    }
    count
}

/// Fast read of all eligible samples from a module (just memcpy, no processing).
/// Call this on the GUI thread with a brief module lock, then pass results
/// to extract_sample_jobs() off-thread.
pub fn read_raw_samples(module: &mut Module) -> Vec<OriginalSample> {
    let num_samples = module.num_samples();
    let mut originals = Vec::new();
    for i in 0..num_samples {
        let rate = module.sample_rate(i);
        let length = module.sample_length_frames(i);
        let channels = module.sample_channels(i);
        if length <= 0 || channels <= 0 || rate >= 48000 {
            continue;
        }
        if let Some(data) = module.read_sample_data(i) {
            let loop_info = module.sample_loop_info(i);
            let loop_prep = LoopPrepPlan::from_sample(length, loop_info);
            let name = module.sample_name(i);
            let bits_per_sample = module.sample_bits_per_sample(i);
            originals.push(OriginalSample {
                index: i,
                data,
                rate,
                channels,
                bits_per_sample,
                source_length_frames: length,
                effective_length_frames: loop_prep.saved_length_frames,
                loop_start_frames: loop_prep.primary_loop_start_frames(),
                looped: loop_prep.is_looped(),
                loop_info,
                name,
            });
        }
    }
    originals
}

/// Process raw samples into jobs ready for AI engines: resample to 48kHz, pad, write WAVs.
/// This is the heavy work — run off the GUI thread.
pub fn extract_sample_jobs(
    raw_samples: &[OriginalSample],
    work_dir: &Path,
    min_duration_secs: f64,
    _cleanup_settings: CleanupSettings,
    cancel_flag: &AtomicBool,
) -> Result<Vec<SampleJob>, String> {
    let mut jobs = Vec::new();

    for o in raw_samples {
        ensure_not_cancelled(cancel_flag)?;
        let i = o.index;
        let rate = o.rate;
        let channels = o.channels;
        let looped = o.looped;

        // DC-strip the original up front so every downstream consumer — the
        // 48 kHz reference, conditioning input, target-RMS math, and the
        // pass-through fallback stored on the job — sees a zero-mean signal.
        // Matches the invariant normalize_sample() enforces on AI outputs so
        // no path leaks DC into the module.
        let mut original_data = o.data.clone();
        remove_dc_per_channel(&mut original_data, channels as usize);

        let resampled_48k =
            build_current_pipeline_reference_48k(&original_data, rate as u32, channels as usize)?;
        let original_length_48k_frames = sample_frame_count(&resampled_48k, channels as usize);

        // Conditioning rate: 24kHz for low-rate samples, 48kHz otherwise.
        // AudioSR requires 24kHz or 48kHz; LavaSR/FLowHigh accept both.
        // Using the lowest valid rate minimizes engine processing time.
        let native_rate = rate as u32;
        let cond_rate: u32 = if native_rate < 24_000 { 24_000 } else { 48_000 };

        // Tile at native rate first — shared across all engines. The
        // conditioning-rate SINC hop (for AudioSR, which requires 24/48 kHz)
        // runs AFTER tiling so every engine sees the same tile boundaries.
        let native_min_samples =
            (min_duration_secs * native_rate as f64).ceil() as usize * channels as usize;
        let (mut native_padded, native_layout) = pad_for_engine(
            &original_data,
            channels as usize,
            native_min_samples,
            o.loop_info,
        );
        normalize_conditioning_input(&mut native_padded, channels as usize);

        // Target RMS for output normalization — compute from DC-removed original
        // so it matches what normalize_sample() does to engine output (DC removal
        // then RMS matching). Without this, DC offset inflates the target and
        // makes upscaled samples louder than the originals.
        let target_rms = dc_free_rms(&original_data, channels as usize);

        ensure_not_cancelled(cancel_flag)?;

        // Write native-rate WAVs (primary input for LavaSR / FLowHigh / AP-BWE).
        let native_inputs_all = write_engine_input_wavs(
            &native_padded,
            channels,
            i,
            work_dir,
            native_rate,
            "_native",
        )?;

        ensure_not_cancelled(cancel_flag)?;

        // AudioSR path: SINC the already-tiled native buffer up to cond_rate.
        // When native == cond, reuse native WAVs directly.
        let (
            conditioning_inputs,
            engine_input_layout,
            native_inputs,
            native_rate_hz,
            native_input_layout,
        ) = if native_rate == cond_rate {
            (native_inputs_all, native_layout, Vec::new(), 0u32, None)
        } else {
            let mut cond_data =
                resample_audio_one_shot(&native_padded, native_rate, cond_rate, channels as usize)?;
            // SINC overshoot (Gibbs ringing at transitions) can push peaks
            // slightly above the pre-SINC normalized level. Re-normalize so
            // AudioSR sees the same peak invariant as before this reorder.
            normalize_conditioning_input(&mut cond_data, channels as usize);
            let cond_inputs =
                write_engine_input_wavs(&cond_data, channels, i, work_dir, cond_rate, "")?;
            let cond_layout = scale_engine_layout(native_layout, native_rate, cond_rate);
            (
                cond_inputs,
                cond_layout,
                native_inputs_all,
                native_rate,
                native_layout,
            )
        };
        ensure_not_cancelled(cancel_flag)?;

        let pcm_sha256 = compute_pcm_sha256(&resampled_48k);

        jobs.push(SampleJob {
            index: i,
            name: o.name.clone(),
            original_data,
            rate,
            output_sample_rate_hz: 48_000,
            channels,
            bits_per_sample: o.bits_per_sample.max(0) as u8,
            source_length_frames: o.source_length_frames,
            looped,
            loop_info: o.loop_info,
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
        });
    }

    Ok(jobs)
}

pub(crate) fn write_engine_input_wavs(
    padded: &[f64],
    channels: i32,
    sample_index: i32,
    work_dir: &Path,
    rate: u32,
    filename_suffix: &str,
) -> Result<Vec<PreparedChannelInput>, String> {
    if channels == 2 {
        let (left, right) = simd::deinterleave_stereo_f64(padded);
        let left_stem = conditioning_stem(sample_index, conditioning_channel_name(channels, 0));
        let right_stem = conditioning_stem(sample_index, conditioning_channel_name(channels, 1));
        let left_path = work_dir.join(format!("{left_stem}{filename_suffix}.wav"));
        write_wav(&left_path, &left, rate, 1)?;
        let right_path = work_dir.join(format!("{right_stem}{filename_suffix}.wav"));
        write_wav(&right_path, &right, rate, 1)?;
        let padded_frames = sample_frame_count(padded, channels as usize);
        Ok(vec![
            PreparedChannelInput {
                channel_index: 0,
                channel_name: conditioning_channel_name(channels, 0).to_string(),
                input_path: left_path,
                input_length_frames: padded_frames,
            },
            PreparedChannelInput {
                channel_index: 1,
                channel_name: conditioning_channel_name(channels, 1).to_string(),
                input_path: right_path,
                input_length_frames: padded_frames,
            },
        ])
    } else {
        let stem = conditioning_stem(sample_index, conditioning_channel_name(channels, 0));
        let input_path = work_dir.join(format!("{stem}{filename_suffix}.wav"));
        write_wav(&input_path, padded, rate, channels as u16)?;
        Ok(vec![PreparedChannelInput {
            channel_index: 0,
            channel_name: conditioning_channel_name(channels, 0).to_string(),
            input_path,
            input_length_frames: sample_frame_count(padded, channels as usize),
        }])
    }
}

/// Scale the rate-dependent offsets inside a layout from one rate to another.
/// `TilingLayout` (single-loop) is rate-agnostic and copied through unchanged;
/// `MixedTilingLayout` stores conditioning-buffer offsets that must be scaled.
pub(crate) fn scale_engine_layout(
    layout: Option<EngineInputLayout>,
    from_rate: u32,
    to_rate: u32,
) -> Option<EngineInputLayout> {
    layout.map(|l| match l {
        EngineInputLayout::Single(t) => EngineInputLayout::Single(t),
        EngineInputLayout::Mixed(m) => EngineInputLayout::Mixed(MixedTilingLayout {
            base_timeline_frames: scaled_frame_count(
                m.base_timeline_frames.max(0) as usize,
                from_rate,
                to_rate,
            ) as i64,
            sustain_block: TiledBlock {
                offset_frames: scaled_frame_count(
                    m.sustain_block.offset_frames.max(0) as usize,
                    from_rate,
                    to_rate,
                ) as i64,
                body_copies: m.sustain_block.body_copies,
            },
            normal_block: TiledBlock {
                offset_frames: scaled_frame_count(
                    m.normal_block.offset_frames.max(0) as usize,
                    from_rate,
                    to_rate,
                ) as i64,
                body_copies: m.normal_block.body_copies,
            },
        }),
        EngineInputLayout::Repeated(r) => EngineInputLayout::Repeated(RepeatedLayout {
            copies: r.copies,
            copy_frames: scaled_frame_count(r.copy_frames, from_rate, to_rate),
        }),
    })
}

fn write_wav(path: &Path, data: &[f64], sample_rate: u32, channels: u16) -> Result<(), String> {
    let wav = build_wav_f64(data, sample_rate, channels);
    std::fs::write(path, wav).map_err(|e| format!("WAV write error: {e}"))
}

fn read_wav(path: &Path) -> Result<(Vec<f64>, u16), String> {
    let wav = std::fs::read(path).map_err(|e| format!("WAV read error: {e}"))?;
    if let Some((data, channels, _)) = parse_wav_f64(&wav) {
        return Ok((data, channels));
    }

    let mut reader = hound::WavReader::open(path).map_err(|e| format!("WAV read error: {e}"))?;
    let spec = reader.spec();
    let channels = spec.channels;
    let data: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map(|v| v as f64))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("WAV decode error: {e}"))?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f64 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("WAV decode error: {e}"))?
        }
    };
    Ok((data, channels))
}

#[cfg(test)]
fn read_wav_header(path: &Path) -> Result<WavHeaderInfo, String> {
    let wav = std::fs::read(path).map_err(|e| format!("WAV read error: {e}"))?;
    parse_wav_header(&wav).ok_or_else(|| "WAV header parse error".to_string())
}

/// Watch a batch output directory for completed WAV files using inotify.
/// The engine creates output files (possibly in subdirectories); each
/// CLOSE_WRITE on a .wav triggers immediate result extraction and delivery.
/// Returns when the child process exits.
enum BatchWatchResult {
    Completed,
    Cancelled,
}

fn watch_batch_outputs(
    child: &mut std::process::Child,
    output_dir: &Path,
    job_indices: &[usize],
    jobs: &[SampleJob],
    engine: &dyn UpsampleEngine,
    ddim_steps: u32,
    engine_label: &str,
    progress_counter: &std::sync::atomic::AtomicI32,
    total_jobs: i32,
    progress_tx: &Sender<RemasterStatus>,
    result_tx: &Sender<RemasterOutput>,
    processed: &mut std::collections::HashSet<usize>,
    pending_outputs: &PendingOutputMap,
    eligible_engine_counts_by_job: &[i32],
    success_counter: &std::sync::atomic::AtomicI32,
    cancel_flag: &AtomicBool,
    progressive: bool,
) -> Result<(), String> {
    use inotify::{Inotify, WatchMask};
    use std::time::Duration;

    // Build a map from per-channel output stem to the owning sample job.
    // Include both hold and release stems for sustain-packed samples so that
    // either output file appearing triggers the extraction attempt.
    let stem_to_job: std::collections::HashMap<String, (usize, usize)> = job_indices
        .iter()
        .flat_map(|&idx| {
            jobs[idx].conditioning_inputs.iter().map(move |input| {
                (
                    conditioning_stem(jobs[idx].index, &input.channel_name),
                    (idx, input.channel_index),
                )
            })
        })
        .collect();
    let mut pending_channels = std::collections::HashMap::<usize, Vec<ChannelResult>>::new();

    let poll_child_until_exit_or_cancel = |child: &mut std::process::Child| -> BatchWatchResult {
        loop {
            if cancellation_requested(cancel_flag) {
                kill_child_best_effort(child);
                return BatchWatchResult::Cancelled;
            }
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => return BatchWatchResult::Completed,
                Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    };

    // Try inotify for real-time output detection; fall back to post-process sweep
    let use_inotify = Inotify::init().ok().and_then(|inotify| {
        inotify
            .watches()
            .add(output_dir, WatchMask::CREATE | WatchMask::CLOSE_WRITE)
            .ok()?;
        Some(inotify)
    });

    if let Some(mut inotify) = use_inotify {
        let mut buf = [0u8; 4096];
        loop {
            if cancellation_requested(cancel_flag) {
                kill_child_best_effort(child);
                return Err(cancelled_error());
            }
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {}
                Err(_) => break,
            }

            let events = match inotify.read_events(&mut buf) {
                Ok(events) => events,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if cancellation_requested(cancel_flag) {
                        kill_child_best_effort(child);
                        return Err(cancelled_error());
                    }
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                Err(_) => break,
            };

            for event in events {
                if cancellation_requested(cancel_flag) {
                    kill_child_best_effort(child);
                    return Err(cancelled_error());
                }
                let Some(name) = event.name.and_then(|n| n.to_str()) else {
                    continue;
                };

                // New subdirectory created — add a watch on it
                if event.mask.contains(inotify::EventMask::CREATE)
                    && event.mask.contains(inotify::EventMask::ISDIR)
                {
                    let subdir = output_dir.join(name);
                    let _ = inotify.watches().add(&subdir, WatchMask::CLOSE_WRITE);
                    continue;
                }

                if !event.mask.contains(inotify::EventMask::CLOSE_WRITE) {
                    continue;
                }
                if !name.ends_with(".wav") {
                    continue;
                }

                // Match the output filename to a sample stem
                let matched_job = stem_to_job.iter().find(|(stem, _)| {
                    name.starts_with(stem.as_str())
                        && name
                            .as_bytes()
                            .get(stem.len())
                            .is_some_and(|&c| c == b'_' || c == b'.')
                });
                let Some((_stem, &(job_idx, channel_index))) = matched_job else {
                    continue;
                };
                if processed.contains(&job_idx) {
                    continue;
                }

                // Brief delay to avoid reading while the engine still holds the file
                std::thread::sleep(Duration::from_millis(1000));
                if cancellation_requested(cancel_flag) {
                    kill_child_best_effort(child);
                    return Err(cancelled_error());
                }

                let job = &jobs[job_idx];
                // Find the hold (main) output WAV for this channel.
                let hold_input = job
                    .conditioning_inputs
                    .iter()
                    .find(|i| i.channel_index == channel_index);
                let hold_wav = hold_input.and_then(|input| {
                    let s = conditioning_stem(job.index, &input.channel_name);
                    engine.find_output_wav(output_dir, &s).ok()
                });
                if let Some(main_path) = hold_wav {
                    match extract_channel_result(job, channel_index, &main_path, None, None) {
                        Ok(channel_result) => {
                            store_channel_result(&mut pending_channels, job_idx, channel_result);
                            let _ = maybe_finish_engine_job(
                                job_idx,
                                jobs,
                                engine.name(),
                                engine.cache_id(),
                                ddim_steps as u16,
                                engine_label,
                                progress_counter,
                                total_jobs,
                                progress_tx,
                                result_tx,
                                processed,
                                pending_outputs,
                                &mut pending_channels,
                                eligible_engine_counts_by_job,
                                success_counter,
                                cancel_flag,
                                progressive,
                            );
                        }
                        Err(e) => {
                            eprintln!("inotify: extract failed for {}: {e}", job.display_name());
                        }
                    }
                }

                if processed.len() == job_indices.len() {
                    return Ok(());
                }
            }
        }
    } else {
        // inotify unavailable — poll the child to completion, then sweep for outputs below
        eprintln!("inotify unavailable, falling back to post-process sweep");
        match poll_child_until_exit_or_cancel(child) {
            BatchWatchResult::Completed => {}
            BatchWatchResult::Cancelled => return Err(cancelled_error()),
        }
    }

    for &job_idx in job_indices {
        if processed.contains(&job_idx) {
            continue;
        }
        let job = &jobs[job_idx];
        for input in &job.conditioning_inputs {
            let already_loaded = pending_channels.get(&job_idx).is_some_and(|results| {
                results
                    .iter()
                    .any(|result| result.channel_index == input.channel_index)
            });
            if already_loaded {
                continue;
            }
            let stem = conditioning_stem(job.index, &input.channel_name);
            match engine.find_output_wav(output_dir, &stem) {
                Ok(wav_path) => {
                    match extract_channel_result(job, input.channel_index, &wav_path, None, None) {
                        Ok(channel_result) => {
                            store_channel_result(&mut pending_channels, job_idx, channel_result);
                        }
                        Err(e) => {
                            eprintln!("sweep: extract failed for {}: {e}", job.display_name());
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{}: no output for {} (stem '{stem}') in {}: {e}",
                        engine.name(),
                        job.display_name(),
                        output_dir.display()
                    );
                }
            }
        }
        let _ = maybe_finish_engine_job(
            job_idx,
            jobs,
            engine.name(),
            engine.cache_id(),
            ddim_steps as u16,
            engine_label,
            progress_counter,
            total_jobs,
            progress_tx,
            result_tx,
            processed,
            pending_outputs,
            &mut pending_channels,
            eligible_engine_counts_by_job,
            success_counter,
            cancel_flag,
            progressive,
        );
    }
    Ok(())
}

/// Length (in output-rate frames) of the equal-power crossfade applied at
/// each stitch seam when the tiled `[head][N×body][tail]` layout is extracted
/// as `[head][middle_body][tail]`. Small fixed window on AI-processed output;
/// no pitch estimation (the output has already been upscaled).
const TILED_SEAM_FADE_FRAMES: usize = 16;

/// Offsets (in output-rate frames) into the engine output for the tiled
/// `[head][N×body][tail]` layout. All values are relative to the start of the
/// raw engine output buffer.
struct TiledOffsets {
    head_frames: usize,
    body_frames: usize,
    tail_frames: usize,
    selected_body_start: usize,
    tail_start: usize,
}

fn tiled_offsets(
    job: &SampleJob,
    layout: TilingLayout,
    trim_frames: usize,
    input_rate_hz: u32,
) -> TiledOffsets {
    tiled_offsets_with_index(
        job,
        layout,
        trim_frames,
        input_rate_hz,
        layout.body_copies / 2,
    )
}

fn tiled_offsets_with_index(
    job: &SampleJob,
    layout: TilingLayout,
    trim_frames: usize,
    input_rate_hz: u32,
    copy_index: usize,
) -> TiledOffsets {
    let output_rate = job.output_sample_rate_hz.max(0) as u32;
    let native_rate = job.rate.max(0) as u32;
    let cond_rate = input_rate_hz;
    let loop_native = match layout.loop_source {
        TiledLoopSource::Normal => job.loop_info.normal,
        TiledLoopSource::Sustain => job.loop_info.sustain,
    };
    // Match the two-step scaling the engine input went through: native→cond
    // rate in `pad_for_engine`, then cond→output rate inside the engine.
    // Computing native→output directly would re-round differently and drift
    // by up to 1 frame per region at odd native rates, compounding across
    // the N body copies.
    let loop_cond = scaled_loop_region(loop_native, native_rate, cond_rate);
    let head_cond = loop_cond.start_frames.max(0) as usize;
    let body_cond = (loop_cond.end_frames - loop_cond.start_frames).max(0) as usize;
    let head = scaled_frame_count(head_cond, cond_rate, output_rate);
    let body = scaled_frame_count(body_cond, cond_rate, output_rate);
    // Compute tail by subtraction so the three independent rate divisions
    // can't compound into an off-by-one mismatch against
    // `original_length_48k_frames`.
    let tail = trim_frames.saturating_sub(head).saturating_sub(body);
    let idx = copy_index.min(layout.body_copies.saturating_sub(1));
    TiledOffsets {
        head_frames: head,
        body_frames: body,
        tail_frames: tail,
        selected_body_start: head + idx * body,
        tail_start: head + layout.body_copies * body,
    }
}

/// Result of FFT cross-correlation based best-copy selection over a tiled
/// loop block.
#[derive(Debug, Clone)]
struct BestCopyResult {
    best_index: usize,
    best_score: f64,
    all_scores: Vec<f64>,
}

/// Minimum number of template frames required to run correlation-based
/// selection. Below this, correlation is too noisy to be reliable.
const BEST_COPY_MIN_TEMPLATE_FRAMES: usize = 64;

/// Noise floor for the peak correlation score. Scores below this indicate
/// the template doesn't match any copy well (e.g., AI output corrupted or
/// template extracted from the wrong region); caller falls back to middle.
///
/// `fft_cross_correlation` returns 1.0 at perfect match and ~0 for
/// uncorrelated content, so this is a fraction-of-perfect-match floor.
/// 0.5 means "the best copy must be at least half as similar to the
/// reference as a perfect copy would be" — which catches mangled AI
/// outputs but accepts the typical 0.85-0.99 range produced by clean
/// upscaling.
const BEST_COPY_SCORE_FLOOR: f64 = 0.5;

/// Select the tiled body copy whose content best matches the original loop
/// body (taken from `job.reference_48k`). Returns `None` when the template
/// is too short or all copies score below the noise floor, in which case
/// the caller should use the middle copy as before.
///
/// `channel_index` selects which channel of the stereo source to score
/// against. For mono engine output of stereo source samples (the standard
/// path), `raw_data` is the single channel, but `reference_48k` is the
/// full interleaved stereo source — so the matching channel is pulled out
/// here.
fn select_best_body_copy(
    raw_data: &[f64],
    new_channels: usize,
    job: &SampleJob,
    layout: TilingLayout,
    trim_frames: usize,
    input_rate_hz: u32,
    channel_index: usize,
) -> Option<BestCopyResult> {
    let ref_channels = job.channels.max(0) as usize;
    if new_channels == 0 || ref_channels == 0 || job.reference_48k.is_empty() {
        return None;
    }

    let off = tiled_offsets(job, layout, trim_frames, input_rate_hz);
    if off.body_frames < BEST_COPY_MIN_TEMPLATE_FRAMES {
        return None;
    }

    // Scale the original loop region into reference_48k coordinates using
    // the same two-step native→cond→output path as tiled_offsets, so the
    // template boundaries stay aligned with the tiled copies in the AI
    // output.
    let output_rate = job.output_sample_rate_hz.max(0) as u32;
    let native_rate = job.rate.max(0) as u32;
    let cond_rate = input_rate_hz;
    let loop_native = match layout.loop_source {
        TiledLoopSource::Normal => job.loop_info.normal,
        TiledLoopSource::Sustain => job.loop_info.sustain,
    };
    let loop_cond = scaled_loop_region(loop_native, native_rate, cond_rate);
    let tmpl_start = scaled_frame_count(
        loop_cond.start_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let tmpl_end = tmpl_start + off.body_frames;

    let ref_frames = job.reference_48k.len() / ref_channels.max(1);
    if tmpl_end > ref_frames {
        return None;
    }

    let is_ping_pong = loop_native.mode == SampleLoopMode::PingPong;

    select_best_body_copy_in_block(
        raw_data,
        new_channels,
        &job.reference_48k,
        ref_channels,
        channel_index,
        tmpl_start,
        tmpl_end,
        0,
        off.head_frames,
        off.body_frames,
        layout.body_copies,
        is_ping_pong,
    )
}

/// Inner selector shared by single-loop and mixed-loop paths. Extracts the
/// template from `ref_data[tmpl_start..tmpl_end]` (frame-indexed), then
/// scores each tiled copy in `raw_data` via FFT cross-correlation.
///
/// `block_offset_frames` is where the tiled block (the `[head][N×body]`
/// region) starts in `raw_data`. `head_frames` is the pre-body head length
/// within that block, so copy `k` starts at
/// `block_offset_frames + head_frames + k * body_frames`.
///
/// `ref_channel_index` picks which channel of the (possibly stereo)
/// reference is correlated against `raw_data`. Stereo source samples are
/// processed as separate L/R jobs, so `raw_data` is mono per call but the
/// reference still has the full source channel count.
///
/// Ping-pong layouts alternate forward/backward copies (F,B,F,B,…); odd-
/// indexed copies are scored against a frame-reversed template.
#[allow(clippy::too_many_arguments)]
fn select_best_body_copy_in_block(
    raw_data: &[f64],
    raw_channels: usize,
    ref_data: &[f64],
    ref_channels: usize,
    ref_channel_index: usize,
    tmpl_start_frames: usize,
    tmpl_end_frames: usize,
    block_offset_frames: usize,
    head_frames: usize,
    body_frames: usize,
    body_copies: usize,
    is_ping_pong: bool,
) -> Option<BestCopyResult> {
    if body_copies == 0 || body_frames < BEST_COPY_MIN_TEMPLATE_FRAMES {
        return None;
    }
    if raw_channels == 0 || ref_channels == 0 {
        return None;
    }
    if ref_channel_index >= ref_channels {
        return None;
    }
    if tmpl_end_frames <= tmpl_start_frames {
        return None;
    }
    let tmpl_frames = tmpl_end_frames - tmpl_start_frames;
    if tmpl_frames < BEST_COPY_MIN_TEMPLATE_FRAMES {
        return None;
    }

    // Validate that we have enough reference frames to slice the template.
    let ref_frames = ref_data.len() / ref_channels.max(1);
    if tmpl_end_frames > ref_frames {
        return None;
    }

    // Pull only the channels we actually need: the matching reference
    // channel and the (typically mono) raw signal. Stereo source samples
    // are processed as per-channel jobs by the engine, so raw_data is
    // already mono — split on raw_channels to handle the rare case of
    // multi-channel engine output.
    let raw_channels_split = split_channels(raw_data, raw_channels);
    let ref_channel_data = extract_single_channel(ref_data, ref_channels, ref_channel_index);

    // Search window: the expected center of each copy ± a small tolerance
    // (sub-pitch-period; AI phase drift is bounded).
    let tolerance = (body_frames / 8).min(32);

    let template_fwd: Vec<f64> = ref_channel_data[tmpl_start_frames..tmpl_end_frames].to_vec();
    let template_bwd: Option<Vec<f64>> = if is_ping_pong {
        let mut b = template_fwd.clone();
        b.reverse();
        Some(b)
    } else {
        None
    };

    // Template energy for NCC normalization. Forward and backward
    // templates have identical energy (only frame order differs), so we
    // compute it once.
    let tmpl_energy_sq = template_fwd.iter().map(|x| x * x).sum::<f64>().max(1e-30);

    // For multi-channel raw output, score each channel against the same
    // reference channel and average. The standard path is mono raw
    // (raw_channels = 1), so this loop runs once.
    let mut per_channel_scores: Vec<Vec<f64>> = Vec::with_capacity(raw_channels);

    for raw_ch in raw_channels_split.iter() {
        let corr_fwd = crate::engine::fft_cross_correlation(&template_fwd, raw_ch);
        let corr_bwd = template_bwd
            .as_ref()
            .map(|t| crate::engine::fft_cross_correlation(t, raw_ch));

        // Prefix sum of signal² for O(1) local energy lookups per copy.
        // prefix_sq[i] = Σ raw_ch[0..i]², length raw_ch.len() + 1.
        let mut prefix_sq = Vec::with_capacity(raw_ch.len() + 1);
        prefix_sq.push(0.0f64);
        let mut acc = 0.0f64;
        for &v in raw_ch.iter() {
            acc += v * v;
            prefix_sq.push(acc);
        }

        let mut scores = Vec::with_capacity(body_copies);
        for k in 0..body_copies {
            let copy_start = block_offset_frames + head_frames + k * body_frames;
            let lo = copy_start.saturating_sub(tolerance);
            let hi = (copy_start + tolerance).min(corr_fwd.len().saturating_sub(1));
            let corr = if is_ping_pong && k % 2 == 1 {
                corr_bwd.as_ref().unwrap_or(&corr_fwd)
            } else {
                &corr_fwd
            };
            if lo > hi || corr.is_empty() {
                scores.push(0.0);
                continue;
            }
            let peak = corr[lo..=hi]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if !peak.is_finite() {
                scores.push(0.0);
                continue;
            }

            // NCC correction: the raw `peak` value out of
            // fft_cross_correlation is (Σ signal · template) / tmpl_energy_sq.
            // True NCC divides by √(local_signal_energy · tmpl_energy_sq),
            // so multiplying by √(tmpl_energy_sq / local_signal_energy)
            // converts the partial-scale output into a proper NCC score.
            // This makes the match amplitude-invariant — an AI engine
            // that outputs at a different gain than the reference still
            // scores ≈ 1.0 when the content matches.
            let copy_end = copy_start + body_frames;
            let local_signal_energy = if copy_end <= raw_ch.len() {
                (prefix_sq[copy_end] - prefix_sq[copy_start]).max(1e-30)
            } else if copy_start < raw_ch.len() {
                (prefix_sq[raw_ch.len()] - prefix_sq[copy_start]).max(1e-30)
            } else {
                1e-30
            };
            let ncc = peak * (tmpl_energy_sq / local_signal_energy).sqrt();
            scores.push(if ncc.is_finite() { ncc } else { 0.0 });
        }
        per_channel_scores.push(scores);
    }

    // Average per-copy scores across channels.
    let mut avg_scores = vec![0.0f64; body_copies];
    for scores in &per_channel_scores {
        for (i, &s) in scores.iter().enumerate() {
            avg_scores[i] += s;
        }
    }
    for s in avg_scores.iter_mut() {
        *s /= per_channel_scores.len().max(1) as f64;
    }

    let (best_index, &best_score) = avg_scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    if best_score < BEST_COPY_SCORE_FLOOR {
        return None;
    }

    Some(BestCopyResult {
        best_index,
        best_score,
        all_scores: avg_scores,
    })
}

/// Split interleaved audio into per-channel mono buffers (frame-indexed).
fn split_channels(data: &[f64], channels: usize) -> Vec<Vec<f64>> {
    if channels <= 1 {
        return vec![data.to_vec()];
    }
    let frames = data.len() / channels;
    let mut out: Vec<Vec<f64>> = (0..channels).map(|_| Vec::with_capacity(frames)).collect();
    for frame in data.chunks_exact(channels) {
        for (ch, &v) in frame.iter().enumerate() {
            out[ch].push(v);
        }
    }
    out
}

/// Extract a single channel from interleaved audio without allocating
/// buffers for the unused channels. Equivalent to
/// `split_channels(data, channels).remove(channel_index)` but skips the
/// other channels' allocations.
fn extract_single_channel(data: &[f64], channels: usize, channel_index: usize) -> Vec<f64> {
    if channels <= 1 {
        return data.to_vec();
    }
    let frames = data.len() / channels;
    let mut out = Vec::with_capacity(frames);
    for frame in data.chunks_exact(channels) {
        if channel_index < frame.len() {
            out.push(frame[channel_index]);
        }
    }
    out
}

/// Equal-gain cosine crossfade centered at `seam_frame_in_out` in `out`.
/// Weights are `a_w = cos²(π/2 · t)` and `b_w = sin²(π/2 · t)` so that
/// `a_w + b_w = 1` for all `t`. This is the right choice when the A and B
/// signals are correlated near the seam (head/body/tail regions all produced
/// by the same AI on adjacent inputs): equal-power fades would overshoot
/// identical values by up to √2 at the midpoint and create a false peak at
/// the seam.
///
/// The fade blends the existing content (which is slice A before the seam,
/// slice B after) with "what the other side would have been" pulled from the
/// raw engine output via `a_end_in_raw` (first frame of A's continuation)
/// and `b_start_in_raw` (B's retrodiction starts at `b_start_in_raw - 1`).
/// Frames outside `out`'s valid range are skipped; reads past `raw`'s bounds
/// fall back to the existing `out` value.
fn apply_seam_crossfade(
    out: &mut [f64],
    channels: usize,
    seam_frame_in_out: usize,
    raw: &[f64],
    a_end_in_raw: usize,
    b_start_in_raw: usize,
    fade_frames: usize,
) {
    if channels == 0 || fade_frames == 0 {
        return;
    }
    let out_frames = out.len() / channels;
    let raw_frames = raw.len() / channels;
    let half = fade_frames / 2;
    let denom = (fade_frames - 1).max(1) as f64;

    for i in 0..fade_frames {
        let frame_signed = seam_frame_in_out as isize - half as isize + i as isize;
        if frame_signed < 0 {
            continue;
        }
        let frame = frame_signed as usize;
        if frame >= out_frames {
            break;
        }

        let t = i as f64 / denom;
        let phase = std::f64::consts::FRAC_PI_2 * t;
        let cos_p = phase.cos();
        let sin_p = phase.sin();
        let a_w = cos_p * cos_p;
        let b_w = sin_p * sin_p;

        for ch in 0..channels {
            let out_idx = frame * channels + ch;
            let out_existing = out[out_idx];
            let (a_val, b_val) = if frame < seam_frame_in_out {
                // Before seam: out[frame] is slice A's value. Synthesise B's
                // would-be sample from the engine output's pre-slice region.
                let offset = seam_frame_in_out - frame;
                let b_raw_signed = b_start_in_raw as isize - offset as isize;
                let b_val = if b_raw_signed >= 0 {
                    let idx = b_raw_signed as usize;
                    if idx < raw_frames {
                        raw[idx * channels + ch]
                    } else {
                        out_existing
                    }
                } else {
                    out_existing
                };
                (out_existing, b_val)
            } else {
                // At or after seam: out[frame] is slice B's value. Synthesise
                // A's would-be sample from the engine output's post-slice
                // continuation.
                let offset = frame - seam_frame_in_out;
                let a_raw_idx = a_end_in_raw + offset;
                let a_val = if a_raw_idx < raw_frames {
                    raw[a_raw_idx * channels + ch]
                } else {
                    out_existing
                };
                (a_val, out_existing)
            };
            out[out_idx] = a_val * a_w + b_val * b_w;
        }
    }
}

/// Normal-loop frames that are NOT inside the sustain span. Returns up to
/// two disjoint half-open ranges in output-rate frames; empty if the normal
/// loop is fully contained in sustain (which should not occur in a Mixed
/// layout — `pad_for_engine` falls back to Single sustain for that case).
fn normal_exclusive_ranges(
    n_start: usize,
    n_end: usize,
    s_start: usize,
    s_end: usize,
) -> Vec<(usize, usize)> {
    if n_end <= n_start {
        return Vec::new();
    }
    // Disjoint (sustain before or after normal, no overlap).
    if s_end <= n_start || n_end <= s_start {
        return vec![(n_start, n_end)];
    }
    // Sustain contains normal — defensive; Mixed layout shouldn't reach here.
    if s_start <= n_start && n_end <= s_end {
        return Vec::new();
    }
    // Otherwise, up to two pieces flanking the sustain span.
    let mut out = Vec::new();
    if n_start < s_start {
        out.push((n_start, s_start.min(n_end)));
    }
    if s_end < n_end {
        out.push((s_end.max(n_start), n_end));
    }
    out
}

/// Extract a channel result from a single-loop `[head][N×body][tail]` raw
/// engine output. Returns a buffer of exactly `trim_frames * new_channels`
/// samples (subject to raw bounds).
fn extract_single_channel_output(
    raw_data: &[f64],
    new_channels: usize,
    job: &SampleJob,
    layout: TilingLayout,
    trim_frames: usize,
    input_rate_hz: u32,
    channel_index: usize,
) -> Vec<f64> {
    let middle_idx = layout.body_copies / 2;
    let best = select_best_body_copy(
        raw_data,
        new_channels,
        job,
        layout,
        trim_frames,
        input_rate_hz,
        channel_index,
    );
    let copy_index = best.as_ref().map(|r| r.best_index).unwrap_or(middle_idx);
    if let Some(ref r) = best {
        if r.best_index != middle_idx {
            let scores_str: Vec<String> = r.all_scores.iter().map(|s| format!("{s:.3}")).collect();
            eprintln!(
                "quinlight: [{}] best-copy {}/{} (xcorr={:.4}, was middle={}, scores=[{}])",
                job.display_name(),
                r.best_index,
                layout.body_copies,
                r.best_score,
                middle_idx,
                scores_str.join(", "),
            );
        }
    }
    let off = tiled_offsets_with_index(job, layout, trim_frames, input_rate_hz, copy_index);
    debug_assert_eq!(
        off.head_frames + off.body_frames + off.tail_frames,
        trim_frames,
        "tiled offsets do not sum to original_length_48k_frames ({} + {} + {} vs {})",
        off.head_frames,
        off.body_frames,
        off.tail_frames,
        trim_frames,
    );

    let raw_frames = raw_data.len() / new_channels.max(1);
    let head_end = off.head_frames.min(raw_frames);
    let mb_start = off.selected_body_start.min(raw_frames);
    let mb_end = (off.selected_body_start + off.body_frames).min(raw_frames);
    let t_start = off.tail_start.min(raw_frames);
    let t_end = (off.tail_start + off.tail_frames).min(raw_frames);

    let mut out = Vec::with_capacity(trim_frames * new_channels);
    out.extend_from_slice(&raw_data[0..head_end * new_channels]);
    out.extend_from_slice(&raw_data[mb_start * new_channels..mb_end * new_channels]);
    out.extend_from_slice(&raw_data[t_start * new_channels..t_end * new_channels]);

    // Seam 1: head → middle_body at out-frame head_end.
    // Seam 2: middle_body → tail at out-frame head_end + (mb_end - mb_start).
    let seam1 = head_end;
    let seam2 = head_end + (mb_end - mb_start);

    apply_seam_crossfade(
        &mut out,
        new_channels,
        seam1,
        raw_data,
        off.head_frames,
        off.selected_body_start,
        TILED_SEAM_FADE_FRAMES,
    );
    apply_seam_crossfade(
        &mut out,
        new_channels,
        seam2,
        raw_data,
        off.selected_body_start + off.body_frames,
        off.tail_start,
        TILED_SEAM_FADE_FRAMES,
    );

    out
}

/// Extract a channel result from a mixed-loop raw engine output. The first
/// `trim_frames` of `raw_data` is the base timeline; the sustain-loop span
/// inside that is replaced with the middle-copy sustain body from the tiled
/// sustain block, and the normal-exclusive frames are replaced with the
/// middle-copy normal body from the tiled normal block. Sustain owns any
/// overlap between the two loop regions. Seam crossfades smooth every
/// replaced boundary.
fn extract_mixed_channel_output(
    raw_data: &[f64],
    new_channels: usize,
    job: &SampleJob,
    layout: MixedTilingLayout,
    trim_frames: usize,
    input_rate_hz: u32,
    channel_index: usize,
) -> Vec<f64> {
    let output_rate = job.output_sample_rate_hz.max(0) as u32;
    let native_rate = job.rate.max(0) as u32;
    let cond_rate = input_rate_hz;

    // Seed the output with the base timeline. The base timeline's length
    // stored on the layout is at conditioning rate; scale it to output rate
    // and clamp to `trim_frames` so independent two-step rounding doesn't let
    // us read into the first tiled block. If `base_timeline_out` is short of
    // `trim_frames` (off-by-one from two-step rounding at odd native rates,
    // or the engine underproduced), we let `out` end short rather than
    // zero-padding — mirrors `extract_single_channel_output`'s behavior.
    let raw_frames = raw_data.len() / new_channels.max(1);
    let base_timeline_out = scaled_frame_count(
        layout.base_timeline_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let base_end = trim_frames.min(base_timeline_out).min(raw_frames);
    let mut out = Vec::with_capacity(base_end * new_channels);
    out.extend_from_slice(&raw_data[..base_end * new_channels]);

    // Scale loop regions native → cond → output, matching the two-step path
    // `tiled_offsets` uses so positions don't drift at odd native rates.
    let sustain_cond = scaled_loop_region(job.loop_info.sustain, native_rate, cond_rate);
    let normal_cond = scaled_loop_region(job.loop_info.normal, native_rate, cond_rate);
    let s_start_out = scaled_frame_count(
        sustain_cond.start_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let s_end_out = scaled_frame_count(
        sustain_cond.end_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let n_start_out = scaled_frame_count(
        normal_cond.start_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let n_end_out = scaled_frame_count(
        normal_cond.end_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let s_body_out = s_end_out.saturating_sub(s_start_out);
    let n_body_out = n_end_out.saturating_sub(n_start_out);

    // Block offsets are stored at conditioning rate.
    let s_block_offset_out = scaled_frame_count(
        layout.sustain_block.offset_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );
    let n_block_offset_out = scaled_frame_count(
        layout.normal_block.offset_frames.max(0) as usize,
        cond_rate,
        output_rate,
    );

    // Default middle body (forward copy in both forward 3+ and ping-pong
    // 4k+1 layouts). We override this per-block via FFT cross-correlation
    // when reference_48k provides a usable template.
    let s_middle_idx = layout.sustain_block.body_copies / 2;
    let n_middle_idx = layout.normal_block.body_copies / 2;
    let ref_channels = job.channels.max(0) as usize;

    let s_best_idx = if s_body_out >= BEST_COPY_MIN_TEMPLATE_FRAMES
        && !job.reference_48k.is_empty()
        && ref_channels > 0
        && channel_index < ref_channels
    {
        select_best_body_copy_in_block(
            raw_data,
            new_channels,
            &job.reference_48k,
            ref_channels,
            channel_index,
            s_start_out,
            s_start_out + s_body_out,
            s_block_offset_out,
            s_start_out,
            s_body_out,
            layout.sustain_block.body_copies,
            job.loop_info.sustain.mode == SampleLoopMode::PingPong,
        )
        .map(|r| r.best_index)
        .unwrap_or(s_middle_idx)
    } else {
        s_middle_idx
    };

    let n_best_idx = if n_body_out >= BEST_COPY_MIN_TEMPLATE_FRAMES
        && !job.reference_48k.is_empty()
        && ref_channels > 0
        && channel_index < ref_channels
    {
        select_best_body_copy_in_block(
            raw_data,
            new_channels,
            &job.reference_48k,
            ref_channels,
            channel_index,
            n_start_out,
            n_start_out + n_body_out,
            n_block_offset_out,
            n_start_out,
            n_body_out,
            layout.normal_block.body_copies,
            job.loop_info.normal.mode == SampleLoopMode::PingPong,
        )
        .map(|r| r.best_index)
        .unwrap_or(n_middle_idx)
    } else {
        n_middle_idx
    };

    let s_middle_body_start = s_block_offset_out + s_start_out + s_best_idx * s_body_out;
    let n_middle_body_start = n_block_offset_out + n_start_out + n_best_idx * n_body_out;

    // Pre-compute normal-exclusive ranges so we can suppress sustain seams
    // that would double-blend with a normal seam at the same position.
    let n_exclusive: Vec<(usize, usize)> = if n_body_out > 0 && n_start_out < trim_frames {
        normal_exclusive_ranges(n_start_out, n_end_out, s_start_out, s_end_out)
    } else {
        Vec::new()
    };
    // Sustain and normal exclusive ranges share a boundary when their seams
    // would both land on the same frame. Skip the sustain seam there; the
    // surviving normal seam handles the transition without an equal-power
    // double blend.
    let sustain_exit_shared = n_exclusive.iter().any(|&(r_start, _)| r_start == s_end_out);
    let sustain_entry_shared = n_exclusive
        .iter()
        .any(|&(_, r_end)| r_end.min(trim_frames) == s_start_out);

    // Replace sustain loop span.
    if s_body_out > 0 && s_start_out < trim_frames {
        replace_span_with_middle_body(
            &mut out,
            new_channels,
            raw_data,
            raw_frames,
            s_start_out,
            s_end_out.min(trim_frames),
            s_middle_body_start,
            !sustain_entry_shared,
            !sustain_exit_shared,
            TILED_SEAM_FADE_FRAMES,
        );
    }

    // Replace normal-loop frames that are NOT inside the sustain span.
    for (r_start, r_end) in n_exclusive {
        let body_offset = r_start.saturating_sub(n_start_out);
        replace_span_with_middle_body(
            &mut out,
            new_channels,
            raw_data,
            raw_frames,
            r_start,
            r_end.min(trim_frames),
            n_middle_body_start + body_offset,
            true,
            true,
            TILED_SEAM_FADE_FRAMES,
        );
    }

    out
}

/// Copy the middle-body region of a tiled block (at `src_start` in raw engine
/// output) into `out[dst_start..dst_end]`, then apply equal-power seam
/// crossfades at each replaced boundary. `apply_start_seam` / `apply_end_seam`
/// suppress seams that would double-blend with a seam already emitted by
/// another span (e.g., shared sustain/normal boundaries in mixed layouts).
/// Safe on short raw output — clamps copy length to available source.
fn replace_span_with_middle_body(
    out: &mut [f64],
    new_channels: usize,
    raw_data: &[f64],
    raw_frames: usize,
    dst_start: usize,
    dst_end: usize,
    src_start: usize,
    apply_start_seam: bool,
    apply_end_seam: bool,
    fade_frames: usize,
) {
    // `out` may be shorter than the caller-declared `dst_end` — the mixed
    // extractor allows short outputs when two-step rate rounding leaves the
    // base timeline a frame below `trim_frames`. Clamp every replacement to
    // the actual buffer length so the slice op can't panic.
    let out_frames = out.len() / new_channels.max(1);
    let dst_end = dst_end.min(out_frames);
    if dst_end <= dst_start {
        return;
    }
    let replace_len = dst_end - dst_start;
    let src_end = (src_start + replace_len).min(raw_frames);
    let actual_len = src_end.saturating_sub(src_start).min(replace_len);
    if actual_len == 0 {
        return;
    }

    let dst = &mut out[dst_start * new_channels..(dst_start + actual_len) * new_channels];
    dst.copy_from_slice(
        &raw_data[src_start * new_channels..(src_start + actual_len) * new_channels],
    );

    if apply_start_seam {
        apply_seam_crossfade(
            out,
            new_channels,
            dst_start,
            raw_data,
            dst_start,
            src_start,
            fade_frames,
        );
    }
    if apply_end_seam {
        apply_seam_crossfade(
            out,
            new_channels,
            dst_start + actual_len,
            raw_data,
            src_start + actual_len,
            dst_end.min(raw_frames),
            fade_frames,
        );
    }
}

/// Extract the best-matching copy from a non-looped tiled engine output.
/// The conditioning buffer was `[copy_0][copy_1]...[copy_{N-1}]`, so each
/// copy starts at `k * copy_frames_out` in the output-rate timeline. FFT
/// cross-correlation against `reference_48k` picks the copy whose content
/// best preserves the original. Falls back to copy 0 if the template is
/// too short or reference_48k is missing.
fn extract_repeated_channel_output(
    raw_data: &[f64],
    new_channels: usize,
    job: &SampleJob,
    layout: RepeatedLayout,
    trim_frames: usize,
    input_rate_hz: u32,
    channel_index: usize,
) -> Vec<f64> {
    let output_rate = job.output_sample_rate_hz.max(0) as u32;
    let cond_rate = input_rate_hz;
    let copy_frames_out = scaled_frame_count(layout.copy_frames, cond_rate, output_rate);
    let raw_frames = raw_data.len() / new_channels.max(1);

    let ref_channels = job.channels.max(0) as usize;
    let copy_index = if copy_frames_out >= BEST_COPY_MIN_TEMPLATE_FRAMES
        && !job.reference_48k.is_empty()
        && ref_channels > 0
        && channel_index < ref_channels
        && layout.copies > 1
    {
        // Match the body length and template length exactly so the
        // selector's score normalization stays consistent across copies.
        let tmpl_end = copy_frames_out.min(job.reference_48k.len() / ref_channels.max(1));
        let body_frames = tmpl_end;
        let best = select_best_body_copy_in_block(
            raw_data,
            new_channels,
            &job.reference_48k,
            ref_channels,
            channel_index,
            0,
            tmpl_end,
            0,
            0,
            body_frames,
            layout.copies,
            false,
        );
        if let Some(ref r) = best {
            if r.best_index != 0 {
                let scores_str: Vec<String> =
                    r.all_scores.iter().map(|s| format!("{s:.3}")).collect();
                eprintln!(
                    "quinlight: [{}] best-copy {}/{} (xcorr={:.4}, repeated, scores=[{}])",
                    job.display_name(),
                    r.best_index,
                    layout.copies,
                    r.best_score,
                    scores_str.join(", "),
                );
            }
        }
        best.map(|r| r.best_index).unwrap_or(0)
    } else {
        0
    };

    let start = copy_index * copy_frames_out;
    let end = (start + trim_frames).min(raw_frames);
    let start = start.min(end);
    let mut out = raw_data[start * new_channels..end * new_channels].to_vec();

    // If we picked a non-first copy, apply a short fade-in to mask any
    // residual smear at the copy boundary in the AI output. Non-looped
    // tiled samples have real discontinuities at every copy seam (the
    // sample's last frame ≠ first frame), so the AI may smear those
    // transitions.
    if copy_index > 0 {
        let out_frames = out.len() / new_channels.max(1);
        let fade_frames = REPEATED_COPY_FADE_FRAMES.min(out_frames);
        let denom = fade_frames.saturating_sub(1).max(1) as f64;
        for i in 0..fade_frames {
            let t = i as f64 / denom;
            // Cosine-ramp: w = sin²(π/2 · t), smoothly 0 → 1.
            let w = (std::f64::consts::FRAC_PI_2 * t).sin().powi(2);
            for ch in 0..new_channels {
                out[i * new_channels + ch] *= w;
            }
        }
    }

    out
}

/// Fade-in length used by `extract_repeated_channel_output` when the
/// chosen copy isn't copy 0. Long enough to hide a few-frame AI smear,
/// short enough to not audibly soften legitimate transients.
const REPEATED_COPY_FADE_FRAMES: usize = 64;

fn extract_channel_result(
    job: &SampleJob,
    channel_index: usize,
    output_wav: &Path,
    _release_output_wav: Option<&Path>,
    _output_resample: Option<(u32, u32)>,
) -> Result<ChannelResult, String> {
    let input_rate_hz = job.engine_input_rate();
    let engine_layout = job.engine_layout_for();
    let (raw_data, new_channels_u16) = read_wav(output_wav)?;
    if new_channels_u16 == 0 {
        return Err(format!(
            "Engine output {} had zero channels",
            output_wav.display()
        ));
    }
    if job.channels == 2 && new_channels_u16 != 1 {
        return Err(format!(
            "Expected mono engine output for stereo sample {}, got {} channels",
            job.display_name(),
            new_channels_u16
        ));
    }
    let new_channels = new_channels_u16 as usize;
    let trim_frames = job.original_length_48k_frames.max(0) as usize;

    let extracted = match engine_layout {
        Some(EngineInputLayout::Single(layout)) => extract_single_channel_output(
            &raw_data,
            new_channels,
            job,
            layout,
            trim_frames,
            input_rate_hz,
            channel_index,
        ),
        Some(EngineInputLayout::Mixed(layout)) => extract_mixed_channel_output(
            &raw_data,
            new_channels,
            job,
            layout,
            trim_frames,
            input_rate_hz,
            channel_index,
        ),
        Some(EngineInputLayout::Repeated(layout)) => extract_repeated_channel_output(
            &raw_data,
            new_channels,
            job,
            layout,
            trim_frames,
            input_rate_hz,
            channel_index,
        ),
        None => {
            let trim_samples = trim_frames * new_channels;
            raw_data[..trim_samples.min(raw_data.len())].to_vec()
        }
    };

    let new_length_frames = extracted.len() as i64 / new_channels as i64;
    Ok(ChannelResult {
        channel_index,
        data: extracted,
        length_frames: new_length_frames,
    })
}

fn assemble_sample_candidate(
    job: &SampleJob,
    engine_name: &str,
    channel_results: &[ChannelResult],
) -> Result<SampleResult, String> {
    if channel_results.is_empty() {
        return Err(format!(
            "No channel results to assemble for {}",
            job.display_name()
        ));
    }

    let mut ordered = channel_results.to_vec();
    ordered.sort_by_key(|result| result.channel_index);

    let result = if job.channels == 2 && ordered.len() == 2 {
        let min_frames = ordered
            .iter()
            .map(|result| result.length_frames.max(0))
            .min()
            .unwrap_or(0);
        let left = trim_sample_result(&ordered[0].data, 1, min_frames);
        let right = trim_sample_result(&ordered[1].data, 1, min_frames);
        let data = simd::interleave_stereo_f64(&left, &right);
        let discovered_loops = if job.looped {
            let discovered = search_all_loops(&data, 2, job.loop_info, job.rate as u32, 48_000);
            log_loop_search_result(
                &job.display_name(),
                engine_name,
                job.loop_info,
                job.rate as u32,
                48_000,
                &discovered,
            );
            Some(discovered)
        } else {
            None
        };
        SampleResult {
            index: job.index,
            data,
            length_frames: min_frames,
            channels: 2,
            sample_rate_hz: job.output_sample_rate_hz,
            engine_name: engine_name.to_string(),
            discovered_loops,
        }
    } else if ordered.len() == 1 {
        let data = ordered[0].data.clone();
        let discovered_loops = if job.looped {
            let discovered = search_all_loops(&data, 1, job.loop_info, job.rate as u32, 48_000);
            log_loop_search_result(
                &job.display_name(),
                engine_name,
                job.loop_info,
                job.rate as u32,
                48_000,
                &discovered,
            );
            Some(discovered)
        } else {
            None
        };
        SampleResult {
            index: job.index,
            data,
            length_frames: ordered[0].length_frames,
            channels: 1,
            sample_rate_hz: job.output_sample_rate_hz,
            engine_name: engine_name.to_string(),
            discovered_loops,
        }
    } else {
        return Err(format!(
            "Unsupported channel assembly for {}: expected 1 or 2 channel results, got {}",
            job.display_name(),
            ordered.len()
        ));
    };

    Ok(result)
}

#[cfg(test)]
fn extract_result(
    job: &SampleJob,
    output_wav: &Path,
    engine_name: &str,
) -> Result<SampleResult, String> {
    extract_result_with_release(job, output_wav, None, engine_name)
}

#[cfg(test)]
fn extract_result_with_release(
    job: &SampleJob,
    output_wav: &Path,
    _release_output_wav: Option<&Path>,
    engine_name: &str,
) -> Result<SampleResult, String> {
    let channel_result = extract_channel_result(job, 0, output_wav, None, None)?;
    assemble_sample_candidate(job, engine_name, &[channel_result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openmpt::Module;
    use std::ffi::OsString;
    use std::process::{Child, Command, Stdio};
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn max_parallel_from_memory_zero_memory_trusts_user() {
        assert_eq!(max_parallel_from_memory(0, 3072, 4), 4);
        assert_eq!(max_parallel_from_memory(0, 3072, 1), 1);
    }

    #[test]
    fn max_parallel_from_memory_low_memory_clamps_to_one() {
        // 1 GB / 3 GB = 0 → clamp to 1, never 0.
        assert_eq!(max_parallel_from_memory(1024, 3072, 4), 1);
        // Even 1 MB available: still admit one engine rather than erroring.
        assert_eq!(max_parallel_from_memory(1, 3072, 4), 1);
    }

    #[test]
    fn max_parallel_from_memory_high_memory_caps_at_num_engines() {
        // 64 GB / 3 GB = 21, but we only have 4 engines.
        assert_eq!(max_parallel_from_memory(65_536, 3072, 4), 4);
    }

    #[test]
    fn max_parallel_from_memory_matches_integer_division() {
        // 9 GB / 3 GB = 3 → admit 3.
        assert_eq!(max_parallel_from_memory(9216, 3072, 4), 3);
        // 12 GB / 3 GB = 4 — saturates at num_engines.
        assert_eq!(max_parallel_from_memory(12_288, 3072, 4), 4);
        // 6 GB / 3 GB = 2 → admit 2.
        assert_eq!(max_parallel_from_memory(6144, 3072, 4), 2);
    }

    #[test]
    fn max_parallel_from_memory_zero_engines_short_circuits() {
        assert_eq!(max_parallel_from_memory(65_536, 3072, 0), 0);
    }

    const BASIC_FIXTURE: &str = "mods/2ND_PM.S3M";
    const XM_FIXTURE: &str = "openmpt/test/test.xm";

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    struct HomeGuard {
        previous_home: Option<OsString>,
    }

    impl HomeGuard {
        fn set(path: &Path) -> Self {
            let previous_home = std::env::var_os("HOME");
            // SAFETY: tests serialize HOME mutations through ENV_LOCK.
            unsafe {
                std::env::set_var("HOME", path);
            }
            Self { previous_home }
        }
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            if let Some(ref previous_home) = self.previous_home {
                // SAFETY: tests serialize HOME mutations through ENV_LOCK.
                unsafe {
                    std::env::set_var("HOME", previous_home);
                }
            } else {
                // SAFETY: tests serialize HOME mutations through ENV_LOCK.
                unsafe {
                    std::env::remove_var("HOME");
                }
            }
        }
    }

    struct DummyEngine {
        name: &'static str,
        cache_id: &'static str,
    }

    impl UpsampleEngine for DummyEngine {
        fn name(&self) -> &str {
            self.name
        }

        fn cache_id(&self) -> &str {
            self.cache_id
        }

        fn output_rate(&self) -> u32 {
            48_000
        }

        fn max_batch_size(&self) -> usize {
            1
        }

        fn min_duration_secs(&self) -> f64 {
            5.12
        }

        fn spawn_batch(
            &self,
            _input_manifest: &Path,
            _output_dir: &Path,
            _device: &str,
            _ddim_steps: u32,
            _cpu_thread_budget: usize,
        ) -> Result<Child, String> {
            unreachable!("cached quinlight test should not spawn subprocesses")
        }

        fn find_output_wav(&self, _output_dir: &Path, _stem: &str) -> Result<PathBuf, String> {
            unreachable!("cached quinlight test should not look for output WAVs")
        }
    }

    struct RateLimitedEngine {
        name: &'static str,
        max_original_rate_hz: u32,
    }

    impl UpsampleEngine for RateLimitedEngine {
        fn name(&self) -> &str {
            self.name
        }

        fn cache_id(&self) -> &str {
            "ratelimited-v0.1"
        }

        fn supports_original_rate(&self, original_rate_hz: u32) -> bool {
            original_rate_hz <= self.max_original_rate_hz
        }

        fn output_rate(&self) -> u32 {
            48_000
        }

        fn max_batch_size(&self) -> usize {
            1
        }

        fn min_duration_secs(&self) -> f64 {
            0.0
        }

        fn spawn_batch(
            &self,
            _input_manifest: &Path,
            _output_dir: &Path,
            _device: &str,
            _ddim_steps: u32,
            _cpu_thread_budget: usize,
        ) -> Result<Child, String> {
            unreachable!("rate-limited test engine should not spawn subprocesses")
        }

        fn find_output_wav(&self, _output_dir: &Path, _stem: &str) -> Result<PathBuf, String> {
            unreachable!("rate-limited test engine should not read output WAVs")
        }
    }

    struct SleepEngine;

    impl UpsampleEngine for SleepEngine {
        fn name(&self) -> &str {
            "AudioSR"
        }

        fn cache_id(&self) -> &str {
            "audiosr-v0.1"
        }

        fn output_rate(&self) -> u32 {
            48_000
        }

        fn max_batch_size(&self) -> usize {
            1
        }

        fn min_duration_secs(&self) -> f64 {
            0.0
        }

        fn spawn_batch(
            &self,
            _input_manifest: &Path,
            _output_dir: &Path,
            _device: &str,
            _ddim_steps: u32,
            _cpu_thread_budget: usize,
        ) -> Result<Child, String> {
            Command::new("sh")
                .arg("-c")
                .arg("sleep 30")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("Failed to spawn sleep engine: {e}"))
        }

        fn find_output_wav(&self, _output_dir: &Path, _stem: &str) -> Result<PathBuf, String> {
            Err("sleep engine does not emit WAVs".into())
        }
    }

    fn sample_job(work_dir: &Path) -> SampleJob {
        let input_path = work_dir.join("sample_0_mono.wav");
        std::fs::write(&input_path, []).expect("Should create dummy input WAV");
        let original_data: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 220.0 * 2.0 * std::f64::consts::PI / 16000.0).sin())
            .collect();
        let reference_48k =
            build_current_pipeline_reference_48k(&original_data, 16_000, 1).unwrap_or_default();
        let pcm_sha256 = compute_pcm_sha256(&reference_48k);
        let original_length_48k_frames = sample_frame_count(&reference_48k, 1);
        SampleJob {
            index: 0,
            name: "Kick".to_string(),
            original_data: original_data.clone(),
            rate: 16_000,
            output_sample_rate_hz: 48_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: 4096,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            conditioning_inputs: vec![PreparedChannelInput {
                channel_index: 0,
                channel_name: "mono".into(),
                input_path,
                input_length_frames: 4096,
            }],
            conditioning_rate_hz: 24_000, // 16kHz sample → 24kHz conditioning
            pcm_sha256,
            target_rms: 0.5,
            reference_48k,
            original_length_48k_frames,
            engine_input_layout: None,
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        }
    }

    fn mono_conditioning_inputs(path: PathBuf, frames: i64) -> Vec<PreparedChannelInput> {
        vec![PreparedChannelInput {
            channel_index: 0,
            channel_name: "mono".into(),
            input_path: path,
            input_length_frames: frames,
        }]
    }

    fn normalize_sample_scalar(result: &mut SampleResult, target_rms: f64) {
        let channels = result.channels as usize;
        if channels == 0 || result.data.is_empty() {
            return;
        }
        let frames = result.data.len() / channels;

        for ch in 0..channels {
            let sum: f64 = result
                .data
                .iter()
                .skip(ch)
                .step_by(channels)
                .map(|&s| s as f64)
                .sum();
            let mean = (sum / frames as f64) as f64;
            for s in result.data.iter_mut().skip(ch).step_by(channels) {
                *s -= mean;
            }
        }

        let upscaled_rms = (result
            .data
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / result.data.len() as f64)
            .sqrt() as f64;
        if upscaled_rms > 1e-10 && target_rms > 1e-10 {
            let gain = target_rms / upscaled_rms;
            for s in &mut result.data {
                *s *= gain;
            }
        }
    }

    fn seam_discontinuity(samples: &[f64], channels: usize) -> f64 {
        if channels == 0 || samples.len() < channels * 2 {
            return 0.0;
        }

        let mut max_gap = 0.0f64;
        for ch in 0..channels {
            let first = samples[ch];
            let last = samples[samples.len() - channels + ch];
            max_gap = max_gap.max((first - last).abs());
        }
        max_gap
    }

    fn forward_loop_seam_discontinuity(
        samples: &[f64],
        channels: usize,
        loop_start_frames: usize,
    ) -> f64 {
        if channels == 0 || samples.len() < channels * 2 {
            return 0.0;
        }

        let start = loop_start_frames.saturating_mul(channels);
        if start + channels > samples.len() {
            return 0.0;
        }

        let mut max_gap = 0.0f64;
        for ch in 0..channels {
            let loop_start = samples[start + ch];
            let loop_end = samples[samples.len() - channels + ch];
            max_gap = max_gap.max((loop_start - loop_end).abs());
        }
        max_gap
    }

    fn adjacent_jump_at(samples: &[f64], channels: usize, frame_index: usize) -> f64 {
        if channels == 0 || frame_index == 0 {
            return 0.0;
        }

        let sample_index = frame_index.saturating_mul(channels);
        if sample_index >= samples.len() {
            return 0.0;
        }

        let prev_index = sample_index.saturating_sub(channels);
        let mut max_gap = 0.0f64;
        for ch in 0..channels {
            max_gap = max_gap.max((samples[sample_index + ch] - samples[prev_index + ch]).abs());
        }
        max_gap
    }

    fn frame_gap_between(
        samples: &[f64],
        channels: usize,
        left_frame: usize,
        right_frame: usize,
    ) -> f64 {
        if channels == 0 {
            return 0.0;
        }

        let left_index = left_frame.saturating_mul(channels);
        let right_index = right_frame.saturating_mul(channels);
        if left_index + channels > samples.len() || right_index + channels > samples.len() {
            return 0.0;
        }

        let mut max_gap = 0.0f64;
        for ch in 0..channels {
            max_gap = max_gap.max((samples[left_index + ch] - samples[right_index + ch]).abs());
        }
        max_gap
    }

    fn transient_intensity(samples: &[f64]) -> f64 {
        if samples.len() < 3 {
            return 0.0;
        }

        let mut total = 0.0f64;
        for i in 1..samples.len() - 1 {
            total += (samples[i + 1] - 2.0 * samples[i] + samples[i - 1]).abs();
        }
        total
    }

    fn frequency_component_amplitude(
        samples: &[f64],
        channels: usize,
        sample_rate_hz: u32,
        freq_hz: f64,
    ) -> f64 {
        if channels == 0 || samples.is_empty() {
            return 0.0;
        }

        let frames = sample_frame_count(samples, channels) as usize;
        if frames == 0 {
            return 0.0;
        }

        let mut strongest = 0.0f64;
        for ch in 0..channels {
            let mut sin_dot = 0.0;
            let mut cos_dot = 0.0;
            for frame in 0..frames {
                let sample = samples[frame * channels + ch];
                let phase =
                    2.0 * std::f64::consts::PI * freq_hz * frame as f64 / sample_rate_hz as f64;
                sin_dot += sample * phase.sin();
                cos_dot += sample * phase.cos();
            }
            let amplitude = 2.0 * (sin_dot * sin_dot + cos_dot * cos_dot).sqrt() / frames as f64;
            strongest = strongest.max(amplitude);
        }
        strongest
    }

    fn max_adjacent_jump_in_range(
        samples: &[f64],
        channels: usize,
        start_frame: usize,
        end_frame: usize,
    ) -> f64 {
        if channels == 0 || end_frame <= start_frame + 1 {
            return 0.0;
        }

        let total_frames = sample_frame_count(samples, channels).max(0) as usize;
        let end = end_frame.min(total_frames);
        let start = start_frame.min(end.saturating_sub(1));
        let mut max_jump = 0.0f64;
        for frame in (start + 1)..end {
            max_jump = max_jump.max(adjacent_jump_at(samples, channels, frame));
        }
        max_jump
    }

    fn cleanup_settings(
        mode: CleanupMode,
        engine_version: CleanupEngineVersion,
    ) -> CleanupSettings {
        CleanupSettings::new(mode, engine_version)
    }

    fn off_settings() -> CleanupSettings {
        CleanupSettings::off()
    }

    fn xm_offset_apply_fixture() -> (Module, Vec<SavedEffectParam>, Vec<f64>, u8) {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mapped_note = (1u8..=120)
            .find(|&note| module.instrument_sample_for_note(0, note) == Some(1))
            .expect("Fixture instrument should map at least one note to sample 2");

        assert!(
            module.pattern_num_rows(0) > 0,
            "XM fixture should have pattern data"
        );
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, mapped_note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_OFFSET));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, 0x10));

        let replacement: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 220.0 * 2.0 * std::f64::consts::PI / 48_000.0).sin())
            .collect();
        let saved_effects = save_effect_params(&module, 1);
        let expected_param = ((0x10u32 * 48_000) / 16_000u32).min(255) as u8;
        (module, saved_effects, replacement, expected_param)
    }

    fn all_public_cleanup_settings() -> [CleanupSettings; 6] {
        CleanupSettings::ALL_ACTIVE
    }

    #[test]
    fn cached_quinlight_streams_candidates_and_emits_final_result() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let job = sample_job(work_dir.path());
        let audio = job.reference_48k.clone();
        let lava: Vec<f64> = audio
            .iter()
            .enumerate()
            .map(|(i, &sample)| sample + 0.01 * (i as f64 * 7.3).sin())
            .collect();

        cache_store(
            &job.pcm_sha256,
            "audiosr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: audio,
                length_frames: 4096,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );
        cache_store(
            &job.pcm_sha256,
            "lavasr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: lava,
                length_frames: 4096,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "LavaSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );

        let engine = RemasterEngine {
            engines: vec![
                Box::new(DummyEngine {
                    name: "AudioSR",
                    cache_id: "audiosr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "LavaSR",
                    cache_id: "lavasr-v0.1",
                }),
            ],
            fallback_engines: Vec::new(),
        };
        let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("Should create temp remaster dir"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                true,
                false,
            )
            .expect("Cached Quinlight remaster should succeed");

        let results: Vec<RemasterOutput> = result_rx.try_iter().collect();
        let statuses: Vec<RemasterStatus> = progress_rx.try_iter().collect();

        let candidate_names: Vec<&str> = results
            .iter()
            .filter_map(|output| match output {
                RemasterOutput::Candidate(result) => Some(result.engine_name.as_str()),
                RemasterOutput::Final(_) => None,
            })
            .collect();
        let finals: Vec<&SampleResult> = results
            .iter()
            .filter_map(|output| match output {
                RemasterOutput::Final(result) => Some(result),
                RemasterOutput::Candidate(_) => None,
            })
            .collect();

        assert_eq!(
            candidate_names.len(),
            2,
            "Each cached engine should stream separately"
        );
        // Engines run in parallel, so arrival order is non-deterministic.
        let mut sorted_names = candidate_names.clone();
        sorted_names.sort();
        assert_eq!(sorted_names, vec!["AudioSR", "LavaSR"]);
        assert_eq!(
            finals.len(),
            1,
            "Final should only be emitted once all engines have completed"
        );
        assert!(
            finals[0].engine_name.starts_with(QUINLIGHT_NAME),
            "Final should be a Quinlight result",
        );

        let engine_progress: Vec<(i32, i32)> = statuses
            .iter()
            .filter_map(|status| match status {
                RemasterStatus::EngineProgress {
                    sample_index,
                    engines_done,
                    engines_total,
                } => Some((*sample_index, *engines_done, *engines_total)),
                _ => None,
            })
            .map(|(_sample_index, engines_done, engines_total)| (engines_done, engines_total))
            .collect();
        assert_eq!(engine_progress, vec![(1, 2), (2, 2)]);

        let processing_updates: Vec<(i32, i32)> = statuses
            .iter()
            .filter_map(|status| match status {
                RemasterStatus::Processing { current, total, .. } => Some((*current, *total)),
                _ => None,
            })
            .collect();
        assert!(
            processing_updates.iter().all(|(_, total)| *total == 2),
            "Progress total should equal the number of eligible engine/sample pairs",
        );
        assert!(
            processing_updates
                .iter()
                .any(|(current, total)| *current == 2 && *total == 2),
            "Progress should count both engine completions",
        );
    }

    fn cache_three_engine_candidates(job: &SampleJob) {
        let audio = job.reference_48k.clone();
        let lava: Vec<f64> = audio
            .iter()
            .enumerate()
            .map(|(i, &sample)| sample + 0.01 * (i as f64 * 7.3).sin())
            .collect();
        let flow: Vec<f64> = audio
            .iter()
            .enumerate()
            .map(|(i, &sample)| sample + 0.01 * (i as f64 * 3.1).cos())
            .collect();

        for (cache_id, engine_name, data) in [
            ("audiosr-v0.1", "AudioSR", audio),
            ("lavasr-v0.1", "LavaSR", lava),
            ("flowhigh-v0.1", "FLowHigh", flow),
        ] {
            cache_store(
                &job.pcm_sha256,
                cache_id,
                50,
                &SampleResult {
                    index: 0,
                    data,
                    length_frames: 4096,
                    channels: 1,
                    sample_rate_hz: 48_000,
                    engine_name: engine_name.to_string(),
                    discovered_loops: None,
                },
                1.0,
            );
        }
    }

    fn three_dummy_engines() -> RemasterEngine {
        RemasterEngine {
            engines: vec![
                Box::new(DummyEngine {
                    name: "AudioSR",
                    cache_id: "audiosr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "LavaSR",
                    cache_id: "lavasr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "FLowHigh",
                    cache_id: "flowhigh-v0.1",
                }),
            ],
            fallback_engines: Vec::new(),
        }
    }

    #[test]
    fn cli_mode_emits_single_final_with_three_cached_engines() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let job = sample_job(work_dir.path());
        cache_three_engine_candidates(&job);

        let engine = three_dummy_engines();
        let (progress_tx, _progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("Should create temp remaster dir"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                false,
                false,
            )
            .expect("CLI-mode Quinlight remaster should succeed");

        let results: Vec<RemasterOutput> = result_rx.try_iter().collect();
        let candidate_count = results
            .iter()
            .filter(|output| matches!(output, RemasterOutput::Candidate(_)))
            .count();
        let final_count = results
            .iter()
            .filter(|output| matches!(output, RemasterOutput::Final(_)))
            .count();

        assert_eq!(
            candidate_count, 3,
            "Each cached engine should stream a Candidate even in CLI mode",
        );
        assert_eq!(
            final_count, 1,
            "In CLI (progressive=false) mode exactly one Final should be emitted per sample",
        );
    }

    #[test]
    fn gui_mode_emits_progressive_finals_with_three_cached_engines() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let job = sample_job(work_dir.path());
        cache_three_engine_candidates(&job);

        let engine = three_dummy_engines();
        let (progress_tx, _progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("Should create temp remaster dir"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                true,
                false,
            )
            .expect("GUI-mode Quinlight remaster should succeed");

        let results: Vec<RemasterOutput> = result_rx.try_iter().collect();
        let candidate_count = results
            .iter()
            .filter(|output| matches!(output, RemasterOutput::Candidate(_)))
            .count();
        let final_count = results
            .iter()
            .filter(|output| matches!(output, RemasterOutput::Final(_)))
            .count();

        assert_eq!(
            candidate_count, 3,
            "Each cached engine should stream a Candidate"
        );
        assert_eq!(
            final_count, 2,
            "In GUI (progressive=true) mode Final should fire when the 2nd candidate lands and again when the 3rd refines the set",
        );
    }

    #[test]
    fn build_quinlight_result_source_guidance_suppresses_shared_in_band_artifact() {
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let job = sample_job(work_dir.path());
        let reference_48k = reference_48k_from_job(&job).expect("reference should resample");
        let frames = reference_48k.len();
        let artifact_freq = 2_500.0;
        let artifact: Vec<f64> = (0..frames)
            .map(|i| {
                0.08 * (2.0 * std::f64::consts::PI * artifact_freq * i as f64 / 48_000.0).sin()
            })
            .collect();
        let engine_a: Vec<f64> = reference_48k
            .iter()
            .zip(artifact.iter())
            .map(|(&sample, &extra)| sample + extra)
            .collect();
        let engine_b: Vec<f64> = reference_48k
            .iter()
            .zip(artifact.iter())
            .enumerate()
            .map(|(i, (&sample, &extra))| sample + extra + 0.005 * (i as f64 * 0.17).sin())
            .collect();
        let candidates = vec![
            SampleResult {
                index: job.index,
                data: engine_a.clone(),
                length_frames: frames as i64,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            SampleResult {
                index: job.index,
                data: engine_b.clone(),
                length_frames: frames as i64,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "LavaSR".to_string(),
                discovered_loops: None,
            },
        ];
        let unguided = select_quinlight_mix(
            &reference_48k,
            1,
            job.rate as u32,
            &[
                ("AudioSR".to_string(), engine_a, frames as i64, 1),
                ("LavaSR".to_string(), engine_b, frames as i64, 1),
            ],
            2,
            false,
        );
        let guided = build_quinlight_result(&job, &candidates, 2)
            .expect("guided Quinlight result should build");

        let artifact_before =
            frequency_component_amplitude(&unguided.data, 1, 48_000, artifact_freq);
        let artifact_after = frequency_component_amplitude(&guided.data, 1, 48_000, artifact_freq);
        let shared_before = frequency_component_amplitude(&unguided.data, 1, 48_000, 220.0);
        let shared_after = frequency_component_amplitude(&guided.data, 1, 48_000, 220.0);

        assert!(
            guided.engine_name.starts_with(QUINLIGHT_NAME),
            "guided output should still be labeled as Quinlight",
        );
        assert!(
            artifact_after < artifact_before * 0.80,
            "source guidance should suppress shared in-band artifacts in the final Quinlight mix \
             (before={artifact_before:.4}, after={artifact_after:.4})",
        );
        assert!(
            shared_after > shared_before * 0.65,
            "source guidance should keep the core reference band intact \
             (before={shared_before:.4}, after={shared_after:.4})",
        );
    }

    #[test]
    fn build_quinlight_result_no_consensus_returns_48k_sinc_fallback() {
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let job = sample_job(work_dir.path());
        let reference_48k = reference_48k_from_job(&job).expect("reference should resample");
        let candidate = SampleResult {
            index: job.index,
            data: reference_48k.clone(),
            length_frames: reference_48k.len() as i64 / job.channels as i64,
            channels: job.channels,
            sample_rate_hz: 48_000,
            engine_name: "AudioSR".to_string(),
            discovered_loops: None,
        };

        let final_result = build_quinlight_result(&job, &[candidate], 1)
            .expect("no-consensus Quinlight result should build");

        // Consensus failed → name is the bare Quinlight tag (no engine codes),
        // but the payload is the 48 kHz SINC reference so the upsample CLI can
        // still write a fallback file.
        assert_eq!(final_result.engine_name, QUINLIGHT_NAME);
        assert!(is_no_consensus_result(&final_result.engine_name));
        assert_eq!(final_result.sample_rate_hz, 48_000);
        assert_eq!(final_result.channels, job.channels);
        assert_eq!(
            final_result.length_frames,
            job.reference_48k.len() as i64 / job.channels as i64,
        );
        assert_eq!(final_result.data, job.reference_48k);
    }

    #[test]
    fn build_quinlight_result_no_consensus_falls_back_to_native_when_reference_empty() {
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let mut job = sample_job(work_dir.path());
        // Simulate an edge case where the 48 kHz SINC reference couldn't be
        // built (e.g. empty source on a pathological input): the no-consensus
        // path must still produce a usable SampleResult, so it falls back to
        // the native-rate original.
        job.reference_48k.clear();
        let candidate = SampleResult {
            index: job.index,
            data: vec![0.0; 4096],
            length_frames: 4096,
            channels: job.channels,
            sample_rate_hz: 48_000,
            engine_name: "AudioSR".to_string(),
            discovered_loops: None,
        };

        let final_result = build_quinlight_result(&job, &[candidate], 1)
            .expect("no-consensus fallback should still build a result");

        assert!(is_no_consensus_result(&final_result.engine_name));
        assert_eq!(final_result.sample_rate_hz, job.rate);
        assert_eq!(final_result.length_frames, job.source_length_frames);
        assert_eq!(final_result.channels, job.channels);
        assert_eq!(final_result.data, job.original_data);
    }

    #[test]
    fn select_quinlight_mix_internal_prefers_native_source_scoring_when_available() {
        let source_rate = 16_000u32;
        let candidate_rate = 48_000u32;
        let source_frames = 4096usize;
        let candidate_frames = scaled_frame_count(source_frames, source_rate, candidate_rate);
        let tone = |sample_rate: u32, frames: usize, freq_hz: f64, amp: f64| -> Vec<f64> {
            (0..frames)
                .map(|i| {
                    let t = i as f64 / sample_rate as f64;
                    amp * (2.0 * std::f64::consts::PI * freq_hz * t).sin()
                })
                .collect()
        };
        let source_reference: Vec<f64> = (0..source_frames)
            .map(|i| {
                let t = i as f64 / source_rate as f64;
                0.9 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 3_200.0 * t + 0.3).sin()
            })
            .collect();
        let reference_48k = tone(candidate_rate, candidate_frames, 440.0, 0.9);
        let source_like: Vec<f64> = (0..candidate_frames)
            .map(|i| {
                let t = i as f64 / candidate_rate as f64;
                0.9 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 3_200.0 * t + 0.3).sin()
            })
            .collect();
        let engines = vec![
            (
                "ReferenceLike".to_string(),
                reference_48k.clone(),
                candidate_frames as i64,
                1,
            ),
            (
                "SourceLike".to_string(),
                source_like,
                candidate_frames as i64,
                1,
            ),
        ];

        let legacy = select_quinlight_mix_internal(
            1,
            source_rate,
            QuinlightSelectionReference {
                target_rms: rms_or_zero(&reference_48k),
                source_native: None,
                score_48k: Some(&reference_48k),
                fallback_48k: Some(&reference_48k),
            },
            &engines,
            engines.len(),
            false,
        );
        let native_guided = select_quinlight_mix_internal(
            1,
            source_rate,
            QuinlightSelectionReference {
                target_rms: rms_or_zero(&source_reference),
                source_native: Some(&source_reference),
                score_48k: Some(&reference_48k),
                fallback_48k: Some(&reference_48k),
            },
            &engines,
            engines.len(),
            false,
        );

        assert_eq!(
            legacy.contributors[0].name, "ReferenceLike",
            "Legacy 48 kHz reference scoring should favor the reference-like candidate"
        );
        assert_eq!(
            native_guided.contributors[0].name, "SourceLike",
            "Native-rate source scoring should favor the source-like candidate"
        );
    }

    #[test]
    fn skips_rate_limited_engine_before_cache_or_dispatch() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let mut job = sample_job(work_dir.path());
        job.rate = 22_050;

        cache_store(
            &job.pcm_sha256,
            "audiosr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: job.reference_48k.clone(),
                length_frames: 4096,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );
        cache_store(
            &job.pcm_sha256,
            "ratelimited-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: job.reference_48k.clone(),
                length_frames: 4096,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "LavaSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );

        let engine = RemasterEngine::from_test_engines(vec![
            Box::new(DummyEngine {
                name: "AudioSR",
                cache_id: "audiosr-v0.1",
            }),
            Box::new(RateLimitedEngine {
                name: "LavaSR",
                max_original_rate_hz: 16_000,
            }),
        ]);
        let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("Should create temp remaster dir"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                true,
                false,
            )
            .expect("rate-gated remaster should succeed");

        let results: Vec<RemasterOutput> = result_rx.try_iter().collect();
        let statuses: Vec<RemasterStatus> = progress_rx.try_iter().collect();

        let candidate_names: Vec<&str> = results
            .iter()
            .filter_map(|output| match output {
                RemasterOutput::Candidate(result) => Some(result.engine_name.as_str()),
                RemasterOutput::Final(_) => None,
            })
            .collect();
        assert_eq!(candidate_names, vec!["AudioSR"]);

        let engine_progress: Vec<(i32, i32)> = statuses
            .iter()
            .filter_map(|status| match status {
                RemasterStatus::EngineProgress {
                    sample_index: _,
                    engines_done,
                    engines_total,
                } => Some((*engines_done, *engines_total)),
                _ => None,
            })
            .collect();
        assert_eq!(engine_progress, vec![(1, 1)]);

        let processing_updates: Vec<(i32, i32)> = statuses
            .iter()
            .filter_map(|status| match status {
                RemasterStatus::Processing { current, total, .. } => Some((*current, *total)),
                _ => None,
            })
            .collect();
        assert!(
            processing_updates.iter().all(|(_, total)| *total == 1),
            "Only AudioSR should contribute to progress totals",
        );
        assert!(
            statuses.iter().any(|status| matches!(
                status,
                RemasterStatus::Log(message)
                    if message.contains("LavaSR skipped")
            )),
            "skip log should be emitted for rate-limited engine",
        );
    }

    #[test]
    fn remaster_samples_completes_cleanly_when_no_engine_supports_original_rate() {
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let mut job = sample_job(work_dir.path());
        job.rate = 22_050;

        let engine = RemasterEngine::from_test_engines(vec![Box::new(RateLimitedEngine {
            name: "LavaSR",
            max_original_rate_hz: 16_000,
        })]);
        let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("Should create temp remaster dir"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                true,
                false,
            )
            .expect("unsupported samples should be skipped cleanly");

        let results: Vec<RemasterOutput> = result_rx.try_iter().collect();
        let statuses: Vec<RemasterStatus> = progress_rx.try_iter().collect();

        assert!(results.is_empty(), "no engine results should be emitted");
        assert!(
            statuses
                .iter()
                .all(|status| !matches!(status, RemasterStatus::EngineProgress { .. })),
            "no engine progress should be emitted when nothing is eligible",
        );
        assert!(
            statuses.iter().any(|status| matches!(
                status,
                RemasterStatus::Log(message)
                    if message.contains("no selected engine supports the original sample rates")
            )),
            "clean skip should explain why no AI work ran",
        );
        assert!(
            matches!(statuses.last(), Some(RemasterStatus::Complete)),
            "remaster should still complete cleanly",
        );
    }

    // Test removed: remaster_samples_lavasr_22050_one_shot_uses_engine_specific_staged_inputs
    // LavaSR-specific conditioning inputs were removed in the pipeline simplification.

    #[test]
    fn extract_sample_jobs_returns_cancelled_when_requested_before_processing() {
        let raw_samples = vec![OriginalSample {
            index: 0,
            data: vec![0.0; 1024],
            rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: 1024,
            effective_length_frames: 1024,
            loop_start_frames: 0,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            name: "Kick".into(),
        }];
        let work_dir = tempfile::tempdir().expect("tempdir");
        let cancel_flag = AtomicBool::new(true);

        match extract_sample_jobs(
            &raw_samples,
            work_dir.path(),
            5.12,
            off_settings(),
            &cancel_flag,
        ) {
            Ok(_) => panic!("cancelled extraction should fail"),
            Err(err) => assert!(is_cancelled_error(&err)),
        }
    }

    #[test]
    fn remaster_samples_returns_cancelled_when_requested_during_engine_work() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("Should create temp work dir");
        let job = sample_job(work_dir.path());
        let engine = RemasterEngine::from_test_engines(vec![Box::new(SleepEngine)]);
        let (progress_tx, _progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = std::sync::Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel_flag.clone();

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            cancel_clone.store(true, Ordering::Relaxed);
        });

        let err = engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("Should create temp remaster dir"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                cancel_flag.as_ref(),
                50,
                true,
                false,
            )
            .expect_err("cancelled remaster should fail");

        assert!(is_cancelled_error(&err));
        assert!(result_rx.try_recv().is_err());
    }

    #[test]
    fn cache_roundtrip_preserves_samples_within_int32_precision() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let dummy_pcm = [0u8; 32];
        let original = vec![-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
        let rms = simd::rms_f64(&original);

        cache_store(
            &dummy_pcm,
            "audiosr-test",
            50,
            &SampleResult {
                index: 3,
                data: original.clone(),
                length_frames: original.len() as i64,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            rms,
        );

        let cached = cache_lookup(&dummy_pcm, "audiosr-test", "", 50, "AudioSR", 3, 48_000)
            .expect("cache entry should exist");

        // 32-bit integer FLAC: quantization error < 1e-9 per sample.
        assert_eq!(cached.data.len(), original.len());
        for (got, want) in cached.data.iter().zip(original.iter()) {
            assert!(
                (got - want).abs() < 1e-8,
                "sample mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn normalize_sample_matches_scalar_reference() {
        let mut actual = SampleResult {
            index: 7,
            data: (0..12_289)
                .flat_map(|i| {
                    let frame = i as f64;
                    [
                        (frame * 0.019).sin() * 0.8 + 0.04,
                        (frame * 0.027).cos() * 0.6 - 0.03,
                    ]
                })
                .collect(),
            length_frames: 12_289,
            channels: 2,
            sample_rate_hz: 48_000,
            engine_name: "AudioSR".to_string(),
            discovered_loops: None,
        };
        let mut expected = actual.clone();

        normalize_sample(&mut actual, 0.37);
        normalize_sample_scalar(&mut expected, 0.37);

        for (actual, expected) in actual.data.iter().zip(expected.data.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "actual={actual}, expected={expected}"
            );
        }
    }

    fn channel_mean(data: &[f64], channels: usize, ch: usize) -> f64 {
        let frames = data.len() / channels;
        let sum: f64 = data.iter().skip(ch).step_by(channels).copied().sum();
        sum / frames as f64
    }

    #[test]
    fn remove_dc_per_channel_zeros_mono_stereo_and_multi() {
        let frames = 8_000usize;

        let mut mono: Vec<f64> = (0..frames)
            .map(|i| 0.5 + 0.3 * (i as f64 * 0.021).sin())
            .collect();
        remove_dc_per_channel(&mut mono, 1);
        assert!(channel_mean(&mono, 1, 0).abs() < 1e-12);

        let mut stereo: Vec<f64> = (0..frames)
            .flat_map(|i| {
                let t = i as f64;
                [
                    0.3 + 0.6 * (t * 0.017).sin(),
                    -0.2 + 0.4 * (t * 0.023).cos(),
                ]
            })
            .collect();
        remove_dc_per_channel(&mut stereo, 2);
        assert!(channel_mean(&stereo, 2, 0).abs() < 1e-12);
        assert!(channel_mean(&stereo, 2, 1).abs() < 1e-12);

        let mut quad: Vec<f64> = (0..frames)
            .flat_map(|i| {
                let t = i as f64;
                [
                    0.10 + 0.5 * (t * 0.011).sin(),
                    -0.15 + 0.4 * (t * 0.019).cos(),
                    0.20 + 0.3 * (t * 0.029).sin(),
                    -0.05 + 0.35 * (t * 0.037).cos(),
                ]
            })
            .collect();
        remove_dc_per_channel(&mut quad, 4);
        for ch in 0..4 {
            assert!(
                channel_mean(&quad, 4, ch).abs() < 1e-12,
                "channel {ch} mean not zero",
            );
        }

        let mut already_clean = mono.clone();
        remove_dc_per_channel(&mut already_clean, 1);
        for (a, b) in already_clean.iter().zip(mono.iter()) {
            assert!((a - b).abs() < 1e-12, "DC removal should be idempotent");
        }
    }

    #[test]
    fn remove_dc_per_channel_handles_sub_frame_remainder() {
        // Stereo with a trailing orphan left sample. Every sample — including the
        // orphan — must have its channel's mean removed; otherwise the last frame
        // carries DC and produces a click at playback.
        let mut stereo_odd: Vec<f64> = vec![
            0.40, -0.10, // frame 0 (L, R)
            0.50, -0.20, // frame 1
            0.60, -0.30, // frame 2
            0.70,  // orphan left
        ];
        let expected_mean_l = (0.40 + 0.50 + 0.60) / 3.0;
        let expected_mean_r = (-0.10 + -0.20 + -0.30) / 3.0;
        remove_dc_per_channel(&mut stereo_odd, 2);

        // Frame-aligned prefix: channel means zero.
        let aligned_sum_l: f64 = stereo_odd[..6].iter().step_by(2).sum();
        let aligned_sum_r: f64 = stereo_odd[..6].iter().skip(1).step_by(2).sum();
        assert!(aligned_sum_l.abs() < 1e-12);
        assert!(aligned_sum_r.abs() < 1e-12);
        // Orphan must have the left-channel mean subtracted, not pass through raw.
        assert!((stereo_odd[6] - (0.70 - expected_mean_l)).abs() < 1e-12);
        // Sanity: we did something — the orphan value changed.
        assert!((stereo_odd[6] - 0.70).abs() > 1e-6);
        let _ = expected_mean_r; // silence unused-warning if asserts are reordered

        // 4-channel with a 2-sample orphan (belongs to channels 0, 1).
        let mut quad_odd = vec![
            0.1, 0.2, 0.3, 0.4, // frame 0
            0.2, 0.3, 0.4, 0.5, // frame 1
            0.3, 0.4, // orphan (ch 0, ch 1)
        ];
        let mean_0 = (0.1 + 0.2) / 2.0;
        let mean_1 = (0.2 + 0.3) / 2.0;
        remove_dc_per_channel(&mut quad_odd, 4);
        // Frame-aligned prefix: all 4 channel means zero.
        for ch in 0..4 {
            let sum: f64 = quad_odd[..8].iter().skip(ch).step_by(4).sum();
            assert!(sum.abs() < 1e-12, "channel {ch} aligned sum not zero");
        }
        // Orphans carry their channel's mean-subtracted value.
        assert!((quad_odd[8] - (0.3 - mean_0)).abs() < 1e-12);
        assert!((quad_odd[9] - (0.4 - mean_1)).abs() < 1e-12);
    }

    #[test]
    fn build_current_pipeline_reference_48k_strips_dc_on_sinc_path() {
        let bias = 0.4_f64;
        let native_rate = 16_000u32;
        let biased: Vec<f64> = (0..4096)
            .map(|i| bias + 0.5 * (i as f64 * 0.013).sin())
            .collect();

        let raw = resample_audio_one_shot(&biased, native_rate, 48_000, 1)
            .expect("raw resample should succeed");
        assert!(
            channel_mean(&raw, 1, 0).abs() > 0.1,
            "sanity: raw resample preserves the DC bias",
        );

        let cleaned = build_current_pipeline_reference_48k(&biased, native_rate, 1)
            .expect("reference build should succeed");
        assert!(
            channel_mean(&cleaned, 1, 0).abs() < 1e-12,
            "SINC-upscaled reference must be zero-mean",
        );
    }

    #[test]
    fn loop_aware_resample_preserves_periodic_loop_seams_at_odd_rates() {
        let input_rate = 8_203u32;
        let frames = 257usize;
        let looped: Vec<f64> = (0..frames)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / frames as f64;
                0.65 * phase.cos() + 0.2 * (phase * 5.0).cos() + 0.1 * (phase * 9.0).sin()
            })
            .collect();

        let one_shot = resample_audio(
            &looped,
            input_rate,
            48_000,
            1,
            ResampleBoundaryMode::OneShot,
        )
        .expect("one-shot resample should succeed");
        let loop_aware = resample_audio(
            &looped,
            input_rate,
            48_000,
            1,
            ResampleBoundaryMode::LoopAware,
        )
        .expect("loop-aware resample should succeed");

        let input_gap = seam_discontinuity(&looped, 1);
        let one_shot_gap = seam_discontinuity(&one_shot, 1);
        let loop_aware_gap = seam_discontinuity(&loop_aware, 1);

        assert!(
            loop_aware_gap <= input_gap + 1.0e-4,
            "loop-aware resample should preserve the original seam continuity (input_gap={input_gap}, loop_aware_gap={loop_aware_gap})",
        );
        assert!(
            loop_aware_gap < one_shot_gap,
            "loop-aware resample should keep a tighter seam than one-shot resampling (one_shot_gap={one_shot_gap}, loop_aware_gap={loop_aware_gap})",
        );
    }

    #[test]
    fn pad_for_engine_always_tiles_already_long_forward_loop() {
        let channels = 1usize;
        let head_frames = 5usize;
        let body_frames = 4usize;
        let tail_frames = 3usize;
        let mut data = Vec::with_capacity(head_frames + body_frames + tail_frames);
        data.resize(head_frames, 0.1);
        data.resize(head_frames + body_frames, 0.5);
        data.resize(head_frames + body_frames + tail_frames, -0.3);

        let loop_info =
            SampleLoopInfo::forward(head_frames as i64, (head_frames + body_frames) as i64);
        let (padded, layout) = pad_for_engine(&data, channels, data.len() / 2, loop_info);
        let layout = layout
            .expect("already-long forward loop should still be tiled")
            .expect_single();

        assert_eq!(layout.body_copies, 3);
        assert_eq!(layout.loop_source, TiledLoopSource::Normal);
        assert_eq!(padded.len(), head_frames + 3 * body_frames + tail_frames);
        assert_eq!(&padded[..head_frames], &data[..head_frames]);
        assert_eq!(
            &padded[head_frames..head_frames + body_frames],
            &data[head_frames..head_frames + body_frames],
        );
        assert_eq!(
            &padded[head_frames + body_frames..head_frames + 2 * body_frames],
            &data[head_frames..head_frames + body_frames],
        );
        assert_eq!(
            &padded[head_frames + 2 * body_frames..head_frames + 3 * body_frames],
            &data[head_frames..head_frames + body_frames],
        );
        assert_eq!(
            &padded[head_frames + 3 * body_frames..],
            &data[head_frames + body_frames..],
        );
    }

    #[test]
    fn pad_for_engine_always_tiles_already_long_pingpong_loop() {
        let channels = 1usize;
        let head_frames = 3usize;
        let body_frames = 5usize;
        let tail_frames = 2usize;
        let mut data = Vec::with_capacity(head_frames + body_frames + tail_frames);
        data.resize(head_frames, 0.0);
        for i in 0..body_frames {
            data.push(i as f64);
        }
        data.resize(head_frames + body_frames + tail_frames, -1.0);

        let loop_info =
            SampleLoopInfo::ping_pong(head_frames as i64, (head_frames + body_frames) as i64);
        let (padded, layout) = pad_for_engine(&data, channels, data.len() / 2, loop_info);
        let layout = layout
            .expect("already-long ping-pong loop should still be tiled")
            .expect_single();
        let forward: Vec<f64> = (0..body_frames).map(|i| i as f64).collect();
        let backward: Vec<f64> = (0..body_frames).rev().map(|i| i as f64).collect();

        assert_eq!(layout.body_copies, 5);
        assert_eq!(padded.len(), head_frames + 5 * body_frames + tail_frames);
        let body_slice = |k: usize| -> &[f64] {
            let start = head_frames + k * body_frames;
            &padded[start..start + body_frames]
        };
        assert_eq!(body_slice(0), forward.as_slice());
        assert_eq!(body_slice(1), backward.as_slice());
        assert_eq!(body_slice(2), forward.as_slice());
        assert_eq!(body_slice(3), backward.as_slice());
        assert_eq!(body_slice(4), forward.as_slice());
        assert_eq!(layout.loop_source, TiledLoopSource::Normal);
    }

    #[test]
    fn pad_for_engine_always_tiles_already_long_sustain_pingpong_loop() {
        let channels = 1usize;
        let head_frames = 4usize;
        let body_frames = 5usize;
        let tail_frames = 3usize;
        let mut data = Vec::with_capacity(head_frames + body_frames + tail_frames);
        data.resize(head_frames, 0.25);
        for i in 0..body_frames {
            data.push(i as f64);
        }
        data.resize(head_frames + body_frames + tail_frames, -0.5);

        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::none(),
            SampleLoopRegion::ping_pong(head_frames as i64, (head_frames + body_frames) as i64),
        );
        let (padded, layout) = pad_for_engine(&data, channels, data.len() / 2, loop_info);
        let layout = layout
            .expect("already-long sustain ping-pong loop should still be tiled")
            .expect_single();
        let forward: Vec<f64> = (0..body_frames).map(|i| i as f64).collect();
        let backward: Vec<f64> = (0..body_frames).rev().map(|i| i as f64).collect();

        assert_eq!(layout.body_copies, 5);
        assert_eq!(layout.loop_source, TiledLoopSource::Sustain);
        let body_slice = |k: usize| -> &[f64] {
            let start = head_frames + k * body_frames;
            &padded[start..start + body_frames]
        };
        assert_eq!(body_slice(0), forward.as_slice());
        assert_eq!(body_slice(1), backward.as_slice());
        assert_eq!(body_slice(2), forward.as_slice());
        assert_eq!(body_slice(3), backward.as_slice());
        assert_eq!(body_slice(4), forward.as_slice());
    }

    #[test]
    fn ping_pong_boundary_layout_reduces_turnaround_jump_at_odd_rates() {
        let input_rate = 8_203u32;
        let attack_len = 17usize;
        let loop_len = 257usize;

        let attack: Vec<f64> = (0..attack_len)
            .map(|i| -0.2 + 0.4 * (i as f64 / attack_len as f64))
            .collect();
        let loop_body: Vec<f64> = (0..loop_len)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / loop_len as f64;
                0.7 * phase.sin() + 0.2 * (phase * 3.0).cos()
            })
            .collect();
        let original: Vec<f64> = attack.iter().chain(loop_body.iter()).copied().collect();
        let loop_info = SampleLoopInfo::ping_pong(attack_len as i64, original.len() as i64);
        let loop_prep = LoopPrepPlan::from_sample(original.len() as i64, loop_info);

        let real_layout =
            build_boundary_loop_input(&original, 1, loop_prep.normal_loop, original.len() + 1024);
        let legacy_layout = build_whole_sample_loop_input(&original, original.len() + 1024);

        let real_resampled =
            resample_audio_one_shot(&real_layout, input_rate, 48_000, 1).expect("real layout");
        let legacy_resampled =
            resample_audio_one_shot(&legacy_layout, input_rate, 48_000, 1).expect("legacy layout");
        let scaled_boundary = scaled_frame_count(original.len(), input_rate, 48_000);

        let real_jump = adjacent_jump_at(&real_resampled, 1, scaled_boundary);
        let legacy_jump = adjacent_jump_at(&legacy_resampled, 1, scaled_boundary);

        assert!(
            real_jump <= legacy_jump + 1e-9,
            "ping-pong boundary layout should reduce the turnaround jump (real_jump={real_jump}, legacy_jump={legacy_jump})",
        );
    }

    #[test]
    fn sustain_forward_reference_preserves_release_tail_and_source_length() {
        let original: Vec<f64> = (0..96)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / 24.0;
                if i < 72 {
                    0.6 * phase.sin()
                } else {
                    0.4 * phase.cos()
                }
            })
            .collect();
        let loop_info =
            SampleLoopInfo::with_loops(SampleLoopRegion::none(), SampleLoopRegion::forward(24, 72));
        let loop_prep = LoopPrepPlan::from_sample(original.len() as i64, loop_info);
        let reference_parts = build_canonical_reference_parts_with_loop_prep(
            &original,
            16_000,
            1,
            loop_prep,
            off_settings(),
        )
        .expect("sustain reference should build");

        assert_eq!(reference_parts.final_reference.len(), original.len());
        assert_eq!(
            reference_parts.hold_reference.as_ref().expect("hold").len(),
            72
        );
        assert_eq!(
            &reference_parts.final_reference[72..],
            &reference_parts.release_reference.as_ref().expect("release")[72..],
            "release tail should be preserved in the final stitched reference",
        );

        let reference_48k = build_quinlight_reference_48k_with_loop_info(
            &original,
            16_000,
            1,
            loop_info,
            off_settings(),
        )
        .expect("48k sustain reference should build");
        assert_eq!(
            reference_48k.len(),
            scaled_frame_count(original.len(), 16_000, 48_000),
        );
    }

    #[test]
    fn sustain_packed_prepare_emits_two_segments_and_stitches_release_tail() {
        let original = vec![0.0, 0.15, 0.4, 0.7, 0.9, 0.8, 0.5, 0.2];
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(5, 8),
            SampleLoopRegion::ping_pong(2, 5),
        );
        let loop_prep = LoopPrepPlan::from_sample(original.len() as i64, loop_info);
        let cancel_flag = AtomicBool::new(false);
        let prepared = prepare_sample_data(
            &original,
            16_000,
            1,
            loop_prep,
            off_settings(),
            0.01,
            &cancel_flag,
        )
        .expect("packed sustain prep should build");

        let PreparedInputLayout::Sustain {
            hold_input_frames,
            release_input_frames,
        } = prepared.input_layout
        else {
            panic!("sustain prep should emit packed input layout");
        };

        assert!(hold_input_frames > 0);
        assert!(release_input_frames > 0);
        assert_eq!(prepared.reference_data.len(), original.len());
        assert_eq!(
            prepared.hold_reference_data.as_ref().expect("hold").len(),
            5
        );
        assert_eq!(
            prepared
                .release_reference_data
                .as_ref()
                .expect("release")
                .len(),
            original.len()
        );
        assert_eq!(
            &prepared.reference_data[5..],
            &prepared.release_reference_data.as_ref().expect("release")[5..],
        );
        assert_eq!(
            prepared.conditioning_input_48k_frames,
            scaled_frame_count(
                sample_frame_count(&prepared.model_input, 1) as usize,
                16_000,
                48_000,
            ) as i64,
            "hold conditioning 48k frames should match scaled hold model input"
        );
        assert!(
            prepared.conditioning_input_48k_release.is_some(),
            "sustain-packed prep should produce a separate release conditioning buffer"
        );
        assert!(
            prepared.conditioning_input_48k_release_frames > 0,
            "release conditioning buffer should have nonzero frame count"
        );
    }

    #[test]
    fn extract_sample_jobs_writes_48khz_float64_conditioning_wav() {
        let original: Vec<f64> = (0..1024)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                0.4 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
            })
            .collect();
        let sample = OriginalSample {
            index: 0,
            data: original.clone(),
            rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: original.len() as i64,
            effective_length_frames: original.len() as i64,
            loop_start_frames: 0,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            name: "Sine".into(),
        };
        let cancel_flag = AtomicBool::new(false);

        let work_dir = tempfile::tempdir().expect("tempdir");
        let jobs = extract_sample_jobs(
            &[sample],
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("job extraction should succeed");
        let job = jobs.first().expect("expected one job");
        let conditioning_input = job
            .conditioning_inputs
            .first()
            .expect("expected one conditioning input");

        let header =
            read_wav_header(&conditioning_input.input_path).expect("conditioning wav should open");
        // 16kHz < 24kHz, so conditioning is resampled to 24kHz
        assert_eq!(
            header.sample_rate, 24_000,
            "conditioning WAV should be at 24kHz"
        );
        assert_eq!(header.channels, 1);
        assert_eq!(header.bits_per_sample, 64);
        assert_eq!(header.sample_format, hound::SampleFormat::Float);

        let (conditioning, channels) =
            read_wav(&conditioning_input.input_path).expect("conditioning wav should decode");
        assert_eq!(channels, 1);

        // 16kHz → 24kHz: scaled_frame_count(1024, 16000, 24000) = 1536 frames.
        // min_samples = ceil(0.01 * 24000) = 240. Since 1536 >= 240, no padding needed.
        let expected_frames = scaled_frame_count(original.len(), 16_000, 24_000) as i64;
        assert_eq!(
            conditioning_input.input_length_frames, expected_frames,
            "input_length_frames should match 24kHz frame count"
        );
        assert_eq!(
            sample_frame_count(&conditioning, channels as usize),
            expected_frames,
            "WAV data frame count should match 24kHz frame count"
        );
    }

    #[test]
    fn extract_sample_jobs_tiles_already_long_forward_loop_conditioning_input() {
        let head_frames = 120usize;
        let body_frames = 120usize;
        let tail_frames = 240usize;
        let mut original = Vec::with_capacity(head_frames + body_frames + tail_frames);
        original.resize(head_frames, 0.1);
        original.resize(head_frames + body_frames, 0.5);
        original.resize(head_frames + body_frames + tail_frames, -0.3);
        let sample = OriginalSample {
            index: 0,
            data: original.clone(),
            rate: 24_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: original.len() as i64,
            effective_length_frames: original.len() as i64,
            loop_start_frames: head_frames as i64,
            looped: true,
            loop_info: SampleLoopInfo::forward(
                head_frames as i64,
                (head_frames + body_frames) as i64,
            ),
            name: "Looped".into(),
        };
        let cancel_flag = AtomicBool::new(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let jobs = extract_sample_jobs(
            &[sample],
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("job extraction should succeed");
        let job = jobs.first().expect("expected one job");
        let conditioning_input = job
            .conditioning_inputs
            .first()
            .expect("expected one conditioning input");
        let layout = job
            .engine_input_layout
            .expect("already-long forward loop should still carry tiling metadata")
            .expect_single();

        assert_eq!(job.conditioning_rate_hz, 48_000);
        assert_eq!(layout.body_copies, 3);
        assert_eq!(layout.loop_source, TiledLoopSource::Normal);
        let expected_head = scaled_frame_count(head_frames, 24_000, 48_000);
        let expected_body = scaled_frame_count(body_frames, 24_000, 48_000);
        let expected_tail = scaled_frame_count(tail_frames, 24_000, 48_000);
        let expected_total = expected_head + 3 * expected_body + expected_tail;
        assert_eq!(
            conditioning_input.input_length_frames,
            expected_total as i64
        );

        let (conditioning, channels) =
            read_wav(&conditioning_input.input_path).expect("conditioning wav should decode");
        assert_eq!(channels, 1);
        assert_eq!(
            sample_frame_count(&conditioning, channels as usize),
            expected_total as i64,
        );

        // Tile identity is an invariant of pad_for_engine at the NATIVE rate.
        // After the SINC hop to conditioning rate, body boundaries pick up
        // kernel-context differences, so identity only holds on the native WAV.
        let native_input = job
            .native_inputs
            .first()
            .expect("native-rate WAV should exist when native_rate != cond_rate");
        let (native_wav, native_channels) =
            read_wav(&native_input.input_path).expect("native wav should decode");
        assert_eq!(native_channels, 1);
        let body_0 = &native_wav[head_frames..head_frames + body_frames];
        let body_1 = &native_wav[head_frames + body_frames..head_frames + 2 * body_frames];
        let body_2 = &native_wav[head_frames + 2 * body_frames..head_frames + 3 * body_frames];
        assert_eq!(body_0, body_1, "first and second body copies should match");
        assert_eq!(body_1, body_2, "second and third body copies should match");
    }

    #[test]
    fn extract_sample_jobs_tiles_already_long_sustain_loop_conditioning_input() {
        let head_frames = 96usize;
        let body_frames = 96usize;
        let tail_frames = 192usize;
        let mut original = Vec::with_capacity(head_frames + body_frames + tail_frames);
        original.resize(head_frames, 0.15);
        original.resize(head_frames + body_frames, 0.45);
        original.resize(head_frames + body_frames + tail_frames, -0.2);
        let sample = OriginalSample {
            index: 0,
            data: original.clone(),
            rate: 24_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: original.len() as i64,
            effective_length_frames: original.len() as i64,
            loop_start_frames: head_frames as i64,
            looped: true,
            loop_info: SampleLoopInfo::with_loops(
                SampleLoopRegion::none(),
                SampleLoopRegion::forward(head_frames as i64, (head_frames + body_frames) as i64),
            ),
            name: "SustainOnly".into(),
        };
        let cancel_flag = AtomicBool::new(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let jobs = extract_sample_jobs(
            &[sample],
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("job extraction should succeed");
        let job = jobs.first().expect("expected one job");
        let layout = job
            .engine_input_layout
            .expect("sustain-only loop should still carry tiling metadata")
            .expect_single();

        assert_eq!(job.conditioning_rate_hz, 48_000);
        assert_eq!(layout.body_copies, 3);
        assert_eq!(layout.loop_source, TiledLoopSource::Sustain);
    }

    #[test]
    fn extract_sample_jobs_splits_stereo_conditioning_into_two_mono_wavs() {
        let original: Vec<f64> = (0..1024)
            .flat_map(|i| {
                let t = i as f64 / 16_000.0;
                [
                    0.4 * (2.0 * std::f64::consts::PI * 330.0 * t).sin(),
                    0.25 * (2.0 * std::f64::consts::PI * 220.0 * t).cos(),
                ]
            })
            .collect();
        let sample = OriginalSample {
            index: 0,
            data: original,
            rate: 16_000,
            channels: 2,
            bits_per_sample: 16,
            source_length_frames: 1024,
            effective_length_frames: 1024,
            loop_start_frames: 0,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            name: "Stereo".into(),
        };
        let cancel_flag = AtomicBool::new(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let jobs = extract_sample_jobs(
            &[sample],
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("job extraction should succeed");
        let job = jobs.first().expect("expected one job");

        assert_eq!(job.conditioning_inputs.len(), 2);
        assert_eq!(job.conditioning_inputs[0].channel_name, "left");
        assert_eq!(job.conditioning_inputs[1].channel_name, "right");

        for input in &job.conditioning_inputs {
            let header = read_wav_header(&input.input_path).expect("conditioning wav");
            assert_eq!(
                header.sample_rate, 24_000,
                "conditioning WAV should be at 24kHz"
            );
            assert_eq!(header.channels, 1);
            assert_eq!(header.bits_per_sample, 64);
            assert_eq!(header.sample_format, hound::SampleFormat::Float);

            let (conditioning, channels) =
                read_wav(&input.input_path).expect("conditioning wav should decode");
            assert_eq!(channels, 1);
            // New pipeline sends raw audio (no normalization to CONDITIONING_TARGET_PEAK).
            // Just verify non-silent data was written.
            assert!(
                simd::peak_abs_f64(&conditioning) > 0.0,
                "conditioning channel should contain non-zero audio data"
            );
        }
        assert_eq!(
            job.conditioning_inputs[0].input_length_frames,
            job.conditioning_inputs[1].input_length_frames
        );

        // Verify left and right channels contain different data (different frequencies).
        let (left_data, _) = read_wav(&job.conditioning_inputs[0].input_path).expect("left wav");
        let (right_data, _) = read_wav(&job.conditioning_inputs[1].input_path).expect("right wav");
        assert_eq!(left_data.len(), right_data.len());
        let differs = left_data
            .iter()
            .zip(right_data.iter())
            .any(|(l, r)| (l - r).abs() > 1.0e-12);
        assert!(
            differs,
            "left and right channels should contain different data"
        );
    }

    #[test]
    fn extract_sample_jobs_hashes_current_prepared_basis_instead_of_raw_pcm() {
        let original: Vec<f64> = (0..1024)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                0.4 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
            })
            .collect();
        let samples = vec![
            OriginalSample {
                index: 0,
                data: original.clone(),
                rate: 16_000,
                channels: 1,
                bits_per_sample: 16,
                source_length_frames: original.len() as i64,
                effective_length_frames: original.len() as i64,
                loop_start_frames: 0,
                looped: false,
                loop_info: SampleLoopInfo::none(),
                name: "16k".into(),
            },
            OriginalSample {
                index: 1,
                data: original.clone(),
                rate: 22_050,
                channels: 1,
                bits_per_sample: 16,
                source_length_frames: original.len() as i64,
                effective_length_frames: original.len() as i64,
                loop_start_frames: 0,
                looped: false,
                loop_info: SampleLoopInfo::none(),
                name: "22k".into(),
            },
        ];
        let raw_hash_a = compute_pcm_sha256(&samples[0].data);
        let raw_hash_b = compute_pcm_sha256(&samples[1].data);
        let cancel_flag = AtomicBool::new(false);
        let work_dir = tempfile::tempdir().expect("tempdir");

        let jobs = extract_sample_jobs(
            &samples,
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("job extraction should succeed");

        assert_eq!(
            raw_hash_a, raw_hash_b,
            "raw PCM is intentionally identical in this regression test",
        );
        assert_eq!(jobs.len(), 2);
        assert_ne!(
            jobs[0].pcm_sha256, raw_hash_a,
            "current cache identity should no longer reuse the raw PCM hash",
        );
        assert_ne!(
            jobs[0].pcm_sha256, jobs[1].pcm_sha256,
            "current cache identity should change when the prepared 48 kHz basis changes",
        );
    }

    // Test removed: extract_sample_jobs_emits_lavasr_16khz_hold_and_release_wavs_for_22050hz_sustain
    // LavaSR-specific 16kHz staging was removed in the pipeline simplification.

    #[test]
    fn write_wav_roundtrips_f64_samples_bit_perfectly() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let wav_path = work_dir.path().join("bit-perfect.wav");
        let original = vec![-1.5, -1.0, -0.125, 0.0, 0.125, 1.0, 1.5];

        write_wav(&wav_path, &original, 48_000, 1).expect("should write f64 bridge wav");

        let header = read_wav_header(&wav_path).expect("header should parse");
        assert_eq!(header.sample_rate, 48_000);
        assert_eq!(header.channels, 1);
        assert_eq!(header.bits_per_sample, 64);
        assert_eq!(header.sample_format, hound::SampleFormat::Float);

        let (decoded, channels) = read_wav(&wav_path).expect("bridge wav should decode");
        assert_eq!(channels, 1);
        assert_eq!(decoded, original);
    }

    #[test]
    fn read_wav_decodes_legacy_float32_files() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let wav_path = work_dir.path().join("legacy-f32.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer =
            hound::WavWriter::create(&wav_path, spec).expect("should create legacy float wav");
        let original = [0.5f32, -0.25, 1.25, -1.5];
        for sample in original {
            writer
                .write_sample(sample)
                .expect("should write legacy float sample");
        }
        writer.finalize().expect("should finalize legacy float wav");

        let (decoded, channels) = read_wav(&wav_path).expect("legacy float wav should decode");
        assert_eq!(channels, 1);
        assert_eq!(
            decoded,
            original
                .into_iter()
                .map(|sample| sample as f64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn read_wav_decodes_legacy_int16_files() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let wav_path = work_dir.path().join("legacy-i16.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::create(&wav_path, spec).expect("should create legacy int wav");
        let original = [0i16, 8192, -16384, 32767];
        for sample in original {
            writer
                .write_sample(sample)
                .expect("should write legacy int sample");
        }
        writer.finalize().expect("should finalize legacy int wav");

        let (decoded, channels) = read_wav(&wav_path).expect("legacy int wav should decode");
        assert_eq!(channels, 1);
        let expected: Vec<f64> = original
            .into_iter()
            .map(|sample| sample as f64 / 32768.0)
            .collect();
        for (actual, expected) in decoded.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn forward_loop_reference_uses_real_loop_boundary_at_odd_rates() {
        let input_rate = 8_203u32;
        let attack_len = 19usize;
        let loop_len = 257usize;
        let tail_len = 11usize;

        let attack: Vec<f64> = (0..attack_len)
            .map(|i| 0.25 * (i as f64 / attack_len as f64))
            .collect();
        let loop_body: Vec<f64> = (0..loop_len)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / loop_len as f64;
                0.6 * phase.sin() + 0.18 * (phase * 3.0).cos()
            })
            .collect();
        let tail = vec![0.95f64; tail_len];
        let original: Vec<f64> = attack
            .iter()
            .chain(loop_body.iter())
            .chain(tail.iter())
            .copied()
            .collect();
        let loop_start = attack_len as i64;
        let loop_end = (attack_len + loop_len) as i64;

        let full_sample_periodic =
            build_quinlight_reference_48k(&original, input_rate, 1, true, off_settings())
                .expect("legacy whole-sample loop-aware reference should build");
        let real_loop_boundary = build_quinlight_reference_48k_with_loop_info(
            &original,
            input_rate,
            1,
            SampleLoopInfo::forward(loop_start, loop_end),
            off_settings(),
        )
        .expect("forward-loop-boundary reference should build");

        let scaled_loop_start = scaled_frame_count(loop_start as usize, input_rate, 48_000);
        let scaled_loop_end = scaled_frame_count(loop_end as usize, input_rate, 48_000);
        let full_gap = forward_loop_seam_discontinuity(&full_sample_periodic, 1, scaled_loop_start);
        let real_gap = forward_loop_seam_discontinuity(&real_loop_boundary, 1, scaled_loop_start);

        assert_eq!(
            real_loop_boundary.len(),
            scaled_loop_end,
            "Forward-loop-boundary reference should truncate at loopEnd after upsampling",
        );
        assert!(
            real_gap < full_gap,
            "Forward-loop-boundary reference should preserve the real loop seam better than repeating the full sample (full_gap={full_gap}, real_gap={real_gap})",
        );
    }

    #[test]
    fn loop_resample_extraction_prefers_middle_full_copy() {
        let channels = 1usize;
        let keep_frames = 4i64;
        let first = [10.0, 11.0, 12.0, 13.0];
        let middle = [20.0, 21.0, 22.0, 23.0];
        let third = [30.0, 31.0, 32.0, 33.0];
        let mut resampled = Vec::new();
        resampled.extend_from_slice(&first);
        resampled.extend_from_slice(&middle);
        resampled.extend_from_slice(&third);

        let extracted = extract_middle_copy_from_loop_resample(&resampled, channels, keep_frames);
        assert_eq!(extracted, &middle[..]);
    }

    #[test]
    fn loop_resample_extraction_uses_later_center_full_copy_when_only_two_copies_fit() {
        let channels = 1usize;
        let keep_frames = 4i64;
        let resampled = vec![10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0];

        let extracted = extract_middle_copy_from_loop_resample(&resampled, channels, keep_frames);
        assert_eq!(extracted, vec![20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn extract_result_without_tiling_trims_from_start() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");
        // Engine output has 8 frames; job expects only 4.
        let engine_output = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        write_wav(&output_wav, &engine_output, 48_000, 1).expect("should write engine output");

        let job = SampleJob {
            index: 0,
            name: "OneShot".into(),
            original_data: vec![0.1, 0.2, 0.3, 0.4, 0.3, 0.4],
            rate: 48_000,
            output_sample_rate_hz: 48_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: 6,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                8,
            ),
            conditioning_rate_hz: 48_000,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: vec![0.1, 0.2, 0.3, 0.4],
            original_length_48k_frames: 4,
            engine_input_layout: None,
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let extracted =
            extract_result(&job, &output_wav, "AudioSR").expect("one-shot extract should work");

        // extract_channel_result trims engine output to original_length_48k_frames
        assert_eq!(extracted.length_frames, 4);
        assert_eq!(extracted.data.len(), 4);
        assert_eq!(
            &extracted.data[..],
            &engine_output[..4],
            "trimmed result must equal first 4 frames of engine output"
        );
    }

    #[test]
    fn pad_for_engine_builds_head_body_tail_layered_layout_for_forward_loop() {
        let channels = 1usize;
        let head_frames = 20usize;
        let body_frames = 30usize;
        let tail_frames = 15usize;
        let (head_val, body_val, tail_val) = (0.1, 0.5, -0.3);
        let mut data = Vec::with_capacity(head_frames + body_frames + tail_frames);
        data.resize(head_frames, head_val);
        data.resize(head_frames + body_frames, body_val);
        data.resize(head_frames + body_frames + tail_frames, tail_val);

        let loop_info =
            SampleLoopInfo::forward(head_frames as i64, (head_frames + body_frames) as i64);
        // Force padding by requesting more than the raw sample length.
        let min_samples = (head_frames + body_frames + tail_frames) * 3;
        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        let layout = layout
            .expect("Forward loop with padding must emit layered layout")
            .expect_single();

        assert!(
            layout.body_copies >= 3,
            "expected at least 3 body copies, got {}",
            layout.body_copies
        );
        assert_eq!(
            padded.len(),
            head_frames + layout.body_copies * body_frames + tail_frames
        );
        // Head region is verbatim.
        assert!(padded[..head_frames].iter().all(|v| *v == head_val));
        // Each body copy holds body_val.
        for copy in 0..layout.body_copies {
            let start = head_frames + copy * body_frames;
            assert!(
                padded[start..start + body_frames]
                    .iter()
                    .all(|v| *v == body_val)
            );
        }
        // Tail appended at the end.
        let tail_start = head_frames + layout.body_copies * body_frames;
        assert!(
            padded[tail_start..tail_start + tail_frames]
                .iter()
                .all(|v| *v == tail_val)
        );
    }

    #[test]
    fn tiled_extraction_reconstructs_head_middle_body_and_tail_from_engine_output() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");

        let rate = 48_000u32;
        let channels = 1usize;
        let head_frames = 32usize;
        let body_frames = 48usize;
        let tail_frames = 32usize;
        let total_frames = head_frames + body_frames + tail_frames;
        let (head_val, body_val, tail_val) = (0.1, 0.5, -0.3);

        let mut data = Vec::with_capacity(total_frames);
        data.resize(head_frames, head_val);
        data.resize(head_frames + body_frames, body_val);
        data.resize(total_frames, tail_val);

        let loop_info =
            SampleLoopInfo::forward(head_frames as i64, (head_frames + body_frames) as i64);
        let min_samples = total_frames * 3;
        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        let layout = layout
            .expect("Forward loop with padding must emit layered layout")
            .expect_single();

        // "Identity engine": write the padded input as the engine output.
        write_wav(&output_wav, &padded, rate, channels as u16).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "Tiled".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total_frames as i64,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: total_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Single(layout)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result =
            extract_result(&job, &output_wav, "AudioSR").expect("tiled extract should succeed");

        assert_eq!(result.length_frames, total_frames as i64);
        assert_eq!(result.data.len(), total_frames);

        // Pick frames well away from either seam (beyond the 16-tap crossfade
        // window). Each region's interior should match the source exactly.
        let fade_half = TILED_SEAM_FADE_FRAMES / 2;
        let head_center = head_frames / 2;
        let body_center = head_frames + body_frames / 2;
        let tail_center = head_frames + body_frames + tail_frames / 2;
        assert!(head_center + fade_half < head_frames);
        assert!(body_center.abs_diff(head_frames) > fade_half);
        assert!(body_center.abs_diff(head_frames + body_frames) > fade_half);
        assert!(tail_center.abs_diff(head_frames + body_frames) > fade_half);

        assert_eq!(result.data[head_center], head_val, "head center");
        assert_eq!(result.data[body_center], body_val, "middle-body center");
        assert_eq!(result.data[tail_center], tail_val, "tail center");

        // Seam fades should keep consecutive-frame jumps bounded — no sharp
        // discontinuities between the stitched regions. With an identity
        // engine the seam-2 body→tail step (body_val=0.5, tail_val=-0.3)
        // has a raw discontinuity of 0.8; the equal-gain crossfade smooths
        // this to under ~half that. Real AI output would produce even
        // smaller seams because the body→tail transition isn't a literal
        // step function in the engine output.
        let raw_discontinuity = (body_val - tail_val).abs();
        let max_step = result
            .data
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_step < raw_discontinuity * 0.65,
            "crossfade should smooth the stitch below 65% of the raw jump \
             ({raw_discontinuity:.3}); max step = {max_step:.4}",
        );
    }

    #[test]
    fn tiled_extraction_reconstructs_already_long_forward_loop_from_middle_body() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");

        let rate = 48_000u32;
        let channels = 1usize;
        let head_frames = 24usize;
        let body_frames = 32usize;
        let tail_frames = 20usize;
        let total_frames = head_frames + body_frames + tail_frames;
        let (head_val, body_val, tail_val) = (0.15, 0.55, -0.25);

        let mut data = Vec::with_capacity(total_frames);
        data.resize(head_frames, head_val);
        data.resize(head_frames + body_frames, body_val);
        data.resize(total_frames, tail_val);

        let loop_info =
            SampleLoopInfo::forward(head_frames as i64, (head_frames + body_frames) as i64);
        let (padded, layout) = pad_for_engine(&data, channels, total_frames / 2, loop_info);
        let layout = layout
            .expect("already-long forward loop should still be tiled")
            .expect_single();
        assert_eq!(layout.body_copies, 3);

        write_wav(&output_wav, &padded, rate, channels as u16).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "TiledLong".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total_frames as i64,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: total_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Single(layout)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result =
            extract_result(&job, &output_wav, "AudioSR").expect("tiled extract should succeed");
        assert_eq!(result.length_frames, total_frames as i64);
        assert_eq!(result.data[head_frames / 2], head_val, "head center");
        assert_eq!(
            result.data[head_frames + body_frames / 2],
            body_val,
            "middle-body center"
        );
        assert_eq!(
            result.data[head_frames + body_frames + tail_frames / 2],
            tail_val,
            "tail center",
        );
    }

    #[test]
    fn tiled_offsets_matches_two_step_engine_scaling_at_odd_native_rate() {
        // Engine path: native 11025 → cond 24000 → output 48000. Layout
        // positions in the 48k engine output are determined by the two-step
        // scaling (pad_for_engine scales native→cond; the engine scales
        // cond→output). A direct native→output scale re-rounds differently
        // and drifts by up to 1 frame per region, compounding across body
        // copies.
        let loop_start_native = 100i64;
        let loop_end_native = 420i64; // body = 320 frames at native rate
        let source_frames_native = 600i64; // tail = 180 frames
        let native_rate = 11_025u32;
        let cond_rate = 24_000u32;
        let output_rate = 48_000u32;

        let job = SampleJob {
            index: 0,
            name: "OddRate".into(),
            original_data: Vec::new(),
            rate: native_rate as i32,
            output_sample_rate_hz: output_rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: source_frames_native,
            looped: true,
            loop_info: SampleLoopInfo::forward(loop_start_native, loop_end_native),
            conditioning_inputs: Vec::new(),
            conditioning_rate_hz: cond_rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: Vec::new(),
            original_length_48k_frames: scaled_frame_count(
                source_frames_native as usize,
                native_rate,
                output_rate,
            ) as i64,
            engine_input_layout: Some(EngineInputLayout::Single(TilingLayout {
                body_copies: 5,
                loop_source: TiledLoopSource::Normal,
            })),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let trim_frames = job.original_length_48k_frames as usize;
        let off = tiled_offsets(
            &job,
            TilingLayout {
                body_copies: 5,
                loop_source: TiledLoopSource::Normal,
            },
            trim_frames,
            job.conditioning_rate_hz,
        );

        // Expected head/body: the same two-step scaling pad_for_engine used.
        let loop_cond = scaled_loop_region(job.loop_info.normal, native_rate, cond_rate);
        let expected_head_cond = loop_cond.start_frames as usize;
        let expected_body_cond = (loop_cond.end_frames - loop_cond.start_frames) as usize;
        let expected_head = scaled_frame_count(expected_head_cond, cond_rate, output_rate);
        let expected_body = scaled_frame_count(expected_body_cond, cond_rate, output_rate);

        assert_eq!(off.head_frames, expected_head);
        assert_eq!(off.body_frames, expected_body);
        // Tail derived by subtraction absorbs all rounding.
        assert_eq!(off.tail_frames, trim_frames - expected_head - expected_body);
        // Middle body is index 2 of 5 copies (forward in FBFBF).
        assert_eq!(off.selected_body_start, expected_head + 2 * expected_body);
        assert_eq!(off.tail_start, expected_head + 5 * expected_body);
    }

    /// Build a mono SampleJob at 48kHz with a synthetic reference and the
    /// given loop info. Used by the best-copy selection tests below.
    fn mono_job_for_best_copy_test(
        reference_48k: Vec<f64>,
        loop_info: SampleLoopInfo,
        body_copies: usize,
    ) -> SampleJob {
        let source_length = reference_48k.len() as i64;
        SampleJob {
            index: 0,
            name: "BestCopyTest".into(),
            original_data: Vec::new(),
            rate: 48_000,
            output_sample_rate_hz: 48_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: source_length,
            looped: loop_info.normal.mode != SampleLoopMode::None,
            loop_info,
            conditioning_inputs: Vec::new(),
            conditioning_rate_hz: 48_000,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k,
            original_length_48k_frames: source_length,
            engine_input_layout: Some(EngineInputLayout::Single(TilingLayout {
                body_copies,
                loop_source: TiledLoopSource::Normal,
            })),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        }
    }

    /// Build a stereo SampleJob with a synthetic interleaved reference. The
    /// reference is interleaved [L0, R0, L1, R1, ...] just like
    /// `build_quinlight_reference_48k` produces in production.
    fn stereo_job_for_best_copy_test(
        reference_48k_interleaved: Vec<f64>,
        loop_info: SampleLoopInfo,
        body_copies: usize,
    ) -> SampleJob {
        let frames = reference_48k_interleaved.len() as i64 / 2;
        SampleJob {
            index: 0,
            name: "StereoBestCopyTest".into(),
            original_data: Vec::new(),
            rate: 48_000,
            output_sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            source_length_frames: frames,
            looped: loop_info.normal.mode != SampleLoopMode::None,
            loop_info,
            conditioning_inputs: Vec::new(),
            conditioning_rate_hz: 48_000,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: reference_48k_interleaved,
            original_length_48k_frames: frames,
            engine_input_layout: Some(EngineInputLayout::Single(TilingLayout {
                body_copies,
                loop_source: TiledLoopSource::Normal,
            })),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        }
    }

    #[test]
    fn select_best_body_copy_prefers_clean_copy() {
        // Reference sample: 400 frames with a loop in [50..250] (body=200).
        let loop_start = 50usize;
        let loop_end = 250usize;
        let total = 400usize;
        let reference: Vec<f64> = (0..total).map(|i| (i as f64 * 0.15).sin()).collect();
        let loop_info = SampleLoopInfo::forward(loop_start as i64, loop_end as i64);

        // Synthetic AI output: [head][5 × body][tail] with copy index 3
        // deliberately corrupted with noise; all others match the reference.
        let head = reference[..loop_start].to_vec();
        let body = reference[loop_start..loop_end].to_vec();
        let tail = reference[loop_end..].to_vec();
        let copies = 5;
        let corrupted_idx = 3;
        let mut raw = Vec::with_capacity(head.len() + copies * body.len() + tail.len());
        raw.extend_from_slice(&head);
        for k in 0..copies {
            if k == corrupted_idx {
                // Replace with noise that doesn't match the reference.
                raw.extend(body.iter().enumerate().map(
                    |(i, _)| {
                        if i % 2 == 0 { 0.5 } else { -0.5 }
                    },
                ));
            } else {
                raw.extend_from_slice(&body);
            }
        }
        raw.extend_from_slice(&tail);

        let job = mono_job_for_best_copy_test(reference, loop_info, copies);
        let layout = TilingLayout {
            body_copies: copies,
            loop_source: TiledLoopSource::Normal,
        };

        let result = select_best_body_copy(&raw, 1, &job, layout, total, 48_000, 0).unwrap();
        assert_ne!(
            result.best_index, corrupted_idx,
            "should not pick the corrupted copy"
        );
        assert!(
            result.all_scores[corrupted_idx] < result.all_scores[result.best_index],
            "corrupted copy score {} should be below winner {}",
            result.all_scores[corrupted_idx],
            result.all_scores[result.best_index],
        );
    }

    #[test]
    fn select_best_body_copy_falls_back_on_empty_reference() {
        let loop_info = SampleLoopInfo::forward(50, 250);
        let layout = TilingLayout {
            body_copies: 5,
            loop_source: TiledLoopSource::Normal,
        };
        let job = mono_job_for_best_copy_test(Vec::new(), loop_info, 5);
        let raw = vec![0.0; 1000];
        assert!(select_best_body_copy(&raw, 1, &job, layout, 400, 48_000, 0).is_none());
    }

    #[test]
    fn select_best_body_copy_falls_back_on_short_template() {
        // Body smaller than BEST_COPY_MIN_TEMPLATE_FRAMES (64) → None.
        let reference: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let loop_info = SampleLoopInfo::forward(20, 60); // body = 40 frames
        let layout = TilingLayout {
            body_copies: 5,
            loop_source: TiledLoopSource::Normal,
        };
        let job = mono_job_for_best_copy_test(reference, loop_info, 5);
        let raw = vec![0.0; 500];
        assert!(select_best_body_copy(&raw, 1, &job, layout, 100, 48_000, 0).is_none());
    }

    #[test]
    fn select_best_body_copy_handles_ping_pong() {
        // Ping-pong layout: F,B,F,B,F. The "corrupted" copy is index 1
        // (a reversed body), so the selector must correctly use the
        // reversed template to score it against the others and still
        // pick one of the forward copies when the reversed one is worse.
        let loop_start = 50usize;
        let loop_end = 250usize;
        let total = 400usize;
        let reference: Vec<f64> = (0..total).map(|i| (i as f64 * 0.2).sin()).collect();
        let mut loop_info = SampleLoopInfo::forward(loop_start as i64, loop_end as i64);
        loop_info.normal.mode = SampleLoopMode::PingPong;

        let head = reference[..loop_start].to_vec();
        let body: Vec<f64> = reference[loop_start..loop_end].to_vec();
        let body_reversed: Vec<f64> = body.iter().copied().rev().collect();
        let tail = reference[loop_end..].to_vec();

        // Build F, corrupted-B, F, B, F. All forward copies should score
        // higher than the corrupted backward copy.
        let copies = 5;
        let mut raw = Vec::new();
        raw.extend_from_slice(&head);
        raw.extend_from_slice(&body); // F
        // Corrupted B: random noise
        raw.extend((0..body.len()).map(|i| if i % 3 == 0 { 0.7 } else { -0.7 }));
        raw.extend_from_slice(&body); // F
        raw.extend_from_slice(&body_reversed); // B
        raw.extend_from_slice(&body); // F
        raw.extend_from_slice(&tail);

        let job = mono_job_for_best_copy_test(reference, loop_info, copies);
        let layout = TilingLayout {
            body_copies: copies,
            loop_source: TiledLoopSource::Normal,
        };
        let result = select_best_body_copy(&raw, 1, &job, layout, total, 48_000, 0).unwrap();
        // Corrupted copy is index 1; should not be picked.
        assert_ne!(result.best_index, 1, "corrupted B copy should not win");
        // The clean B (index 3) should score at least as well as F-copies.
        let score_clean_b = result.all_scores[3];
        let score_corrupted_b = result.all_scores[1];
        assert!(
            score_clean_b > score_corrupted_b,
            "clean backward copy score {} should exceed corrupted backward copy {}",
            score_clean_b,
            score_corrupted_b,
        );
    }

    #[test]
    fn select_best_body_copy_score_is_invariant_under_matched_gain() {
        // When template and signal are scaled by the same factor, scores
        // should be identical. This was historically the only amplitude
        // test; it passes even with the old partial normalization because
        // the gain cancels on both sides. Kept as a regression test for
        // the matched-gain path; see
        // `select_best_body_copy_score_survives_engine_gain_change` for
        // the un-matched case that actually probes the NCC correction.
        let loop_start = 50usize;
        let loop_end = 250usize;
        let total = 400usize;
        let copies = 5;
        let corrupted_idx = 2;

        let build = |amplitude: f64| -> (Vec<f64>, Vec<f64>) {
            let reference: Vec<f64> = (0..total)
                .map(|i| amplitude * (i as f64 * 0.15).sin())
                .collect();
            let head = reference[..loop_start].to_vec();
            let body = reference[loop_start..loop_end].to_vec();
            let tail = reference[loop_end..].to_vec();
            let mut raw = Vec::with_capacity(head.len() + copies * body.len() + tail.len());
            raw.extend_from_slice(&head);
            for k in 0..copies {
                if k == corrupted_idx {
                    raw.extend(
                        (0..body.len()).map(|i| if i % 2 == 0 { amplitude } else { -amplitude }),
                    );
                } else {
                    raw.extend_from_slice(&body);
                }
            }
            raw.extend_from_slice(&tail);
            (reference, raw)
        };

        let loop_info = SampleLoopInfo::forward(loop_start as i64, loop_end as i64);
        let layout = TilingLayout {
            body_copies: copies,
            loop_source: TiledLoopSource::Normal,
        };

        let (loud_ref, loud_raw) = build(1.0);
        let (quiet_ref, quiet_raw) = build(0.001);

        let job_loud = mono_job_for_best_copy_test(loud_ref, loop_info, copies);
        let job_quiet = mono_job_for_best_copy_test(quiet_ref, loop_info, copies);

        let res_loud =
            select_best_body_copy(&loud_raw, 1, &job_loud, layout, total, 48_000, 0).unwrap();
        let res_quiet =
            select_best_body_copy(&quiet_raw, 1, &job_quiet, layout, total, 48_000, 0).unwrap();

        // Same content shape → same scores within numerical precision.
        for (sl, sq) in res_loud.all_scores.iter().zip(res_quiet.all_scores.iter()) {
            assert!(
                (sl - sq).abs() < 1e-6,
                "amplitude should not change normalized scores: loud={sl} quiet={sq}"
            );
        }
        // Both should pick the same (non-corrupted) winner.
        assert_eq!(res_loud.best_index, res_quiet.best_index);
        // And both should be well above the noise floor.
        assert!(
            res_loud.best_score > 0.9,
            "loud sample best score should be near 1.0, got {}",
            res_loud.best_score
        );
        assert!(
            res_quiet.best_score > 0.9,
            "quiet sample best score should be near 1.0, got {}",
            res_quiet.best_score
        );
    }

    #[test]
    fn select_best_body_copy_score_survives_engine_gain_change() {
        // An AI engine that outputs at a different gain than the input
        // would, with only template-energy normalization, score ≈ g at
        // perfect match and fall below the 0.5 score floor for g < 0.5 —
        // silent feature degradation. Proper NCC correction in
        // `select_best_body_copy_in_block` divides each peak by the
        // local signal energy, so the score stays near 1.0 regardless.
        let loop_start = 50usize;
        let loop_end = 250usize;
        let total = 400usize;
        let copies = 5;
        let corrupted_idx = 2;
        let engine_gain = 0.1_f64; // AI output attenuated 10× vs reference

        // Reference at full scale.
        let reference: Vec<f64> = (0..total).map(|i| (i as f64 * 0.15).sin()).collect();

        // Raw engine output: same shape × engine_gain, with one copy
        // corrupted. Corrupted copy uses engine_gain amplitude too so
        // local signal energy is comparable across copies — otherwise
        // the NCC normalization would itself upweight a quiet-but-
        // different copy.
        let head: Vec<f64> = reference[..loop_start]
            .iter()
            .map(|v| v * engine_gain)
            .collect();
        let body: Vec<f64> = reference[loop_start..loop_end]
            .iter()
            .map(|v| v * engine_gain)
            .collect();
        let tail: Vec<f64> = reference[loop_end..]
            .iter()
            .map(|v| v * engine_gain)
            .collect();
        let mut raw = Vec::with_capacity(head.len() + copies * body.len() + tail.len());
        raw.extend_from_slice(&head);
        for k in 0..copies {
            if k == corrupted_idx {
                raw.extend((0..body.len()).map(|i| {
                    if i % 2 == 0 {
                        engine_gain
                    } else {
                        -engine_gain
                    }
                }));
            } else {
                raw.extend_from_slice(&body);
            }
        }
        raw.extend_from_slice(&tail);

        let loop_info = SampleLoopInfo::forward(loop_start as i64, loop_end as i64);
        let layout = TilingLayout {
            body_copies: copies,
            loop_source: TiledLoopSource::Normal,
        };
        let job = mono_job_for_best_copy_test(reference, loop_info, copies);
        let res = select_best_body_copy(&raw, 1, &job, layout, total, 48_000, 0).unwrap();

        // After NCC correction, the gain-attenuated perfect match should
        // score near 1.0, well above the 0.5 score floor.
        assert!(
            res.best_score > 0.9,
            "gain-attenuated perfect match should score near 1.0 after NCC \
             correction, got {} (would be ≈{} without correction)",
            res.best_score,
            engine_gain,
        );
        assert!(
            res.best_score > BEST_COPY_SCORE_FLOOR,
            "score {} must exceed floor {} or the selector will fall back \
             to middle for no content reason",
            res.best_score,
            BEST_COPY_SCORE_FLOOR,
        );
        assert_ne!(
            res.best_index, corrupted_idx,
            "corrupted copy should not win even under gain change"
        );
    }

    #[test]
    fn select_best_body_copy_uses_correct_stereo_channel() {
        // Build a stereo reference where L and R are deliberately different
        // signals (different frequencies). The AI engine emits mono per
        // channel, so the selector must score each channel job against the
        // matching reference channel — not always against L.
        let loop_start = 50usize;
        let loop_end = 250usize;
        let total = 400usize;
        let copies = 5;
        let corrupted_idx_l = 1;
        let corrupted_idx_r = 4; // different copy corrupted in each channel

        // Distinct L/R signals.
        let left: Vec<f64> = (0..total).map(|i| (i as f64 * 0.10).sin()).collect();
        let right: Vec<f64> = (0..total).map(|i| (i as f64 * 0.27).cos()).collect();

        // Interleave.
        let mut interleaved = Vec::with_capacity(total * 2);
        for i in 0..total {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }

        // Build per-channel mono raw outputs (AI returns mono per channel
        // job). For each channel, corrupt a different copy so we can verify
        // that channel 0 → picks against L, channel 1 → picks against R.
        let build_raw = |ch: &Vec<f64>, corrupted_idx: usize| -> Vec<f64> {
            let head = ch[..loop_start].to_vec();
            let body: Vec<f64> = ch[loop_start..loop_end].to_vec();
            let tail = ch[loop_end..].to_vec();
            let mut raw = Vec::with_capacity(head.len() + copies * body.len() + tail.len());
            raw.extend_from_slice(&head);
            for k in 0..copies {
                if k == corrupted_idx {
                    raw.extend((0..body.len()).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }));
                } else {
                    raw.extend_from_slice(&body);
                }
            }
            raw.extend_from_slice(&tail);
            raw
        };
        let raw_left = build_raw(&left, corrupted_idx_l);
        let raw_right = build_raw(&right, corrupted_idx_r);

        let loop_info = SampleLoopInfo::forward(loop_start as i64, loop_end as i64);
        let job = stereo_job_for_best_copy_test(interleaved, loop_info, copies);
        let layout = TilingLayout {
            body_copies: copies,
            loop_source: TiledLoopSource::Normal,
        };

        // Channel 0 (left): correlates raw_left against reference's L
        // channel. Should NOT pick corrupted_idx_l. If we incorrectly
        // pulled R, the test would still pass by accident — so also assert
        // the selector AVOIDS corrupted_idx_l specifically.
        let result_l = select_best_body_copy(&raw_left, 1, &job, layout, total, 48_000, 0).unwrap();
        assert_ne!(
            result_l.best_index, corrupted_idx_l,
            "left channel should avoid its corrupted copy {}",
            corrupted_idx_l,
        );

        // Channel 1 (right): correlates raw_right against reference's R
        // channel. Should NOT pick corrupted_idx_r. Critically: R is a
        // *different* signal from L, so if the selector wrongly used L's
        // reference channel here it would either score badly across the
        // board (and fall back to middle = 2) or pick by accident — but
        // either way it would not consistently identify R's specific
        // corrupted copy at index 4.
        let result_r =
            select_best_body_copy(&raw_right, 1, &job, layout, total, 48_000, 1).unwrap();
        assert_ne!(
            result_r.best_index, corrupted_idx_r,
            "right channel should avoid its corrupted copy {}",
            corrupted_idx_r,
        );
        // The R-channel correlation should also yield high peaks (~1.0)
        // when matched against the right reference, proving that the
        // template wasn't extracted from the wrong channel.
        assert!(
            result_r.best_score > 0.9,
            "right channel best score {} should be near 1.0 when correlated \
             against the matching R reference channel — a low score indicates \
             the wrong reference channel was used",
            result_r.best_score,
        );
    }

    #[test]
    fn tiled_offsets_with_index_picks_selected_copy() {
        let loop_start_native = 100i64;
        let loop_end_native = 420i64;
        let native_rate = 48_000u32;
        let cond_rate = 48_000u32;
        let output_rate = 48_000u32;
        let source_frames_native = 600i64;

        let job = SampleJob {
            index: 0,
            name: "IdxTest".into(),
            original_data: Vec::new(),
            rate: native_rate as i32,
            output_sample_rate_hz: output_rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: source_frames_native,
            looped: true,
            loop_info: SampleLoopInfo::forward(loop_start_native, loop_end_native),
            conditioning_inputs: Vec::new(),
            conditioning_rate_hz: cond_rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: Vec::new(),
            original_length_48k_frames: source_frames_native,
            engine_input_layout: Some(EngineInputLayout::Single(TilingLayout {
                body_copies: 5,
                loop_source: TiledLoopSource::Normal,
            })),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };
        let layout = TilingLayout {
            body_copies: 5,
            loop_source: TiledLoopSource::Normal,
        };
        let trim_frames = source_frames_native as usize;

        let head = 100usize;
        let body = 320usize;

        // Default middle (index 2).
        let off_mid = tiled_offsets(&job, layout, trim_frames, cond_rate);
        assert_eq!(off_mid.selected_body_start, head + 2 * body);

        // Explicit index 0, 3, 4 — tail_start unchanged (always N copies past
        // head).
        let off0 = tiled_offsets_with_index(&job, layout, trim_frames, cond_rate, 0);
        assert_eq!(off0.selected_body_start, head);
        let off3 = tiled_offsets_with_index(&job, layout, trim_frames, cond_rate, 3);
        assert_eq!(off3.selected_body_start, head + 3 * body);
        let off4 = tiled_offsets_with_index(&job, layout, trim_frames, cond_rate, 4);
        assert_eq!(off4.selected_body_start, head + 4 * body);

        // Out-of-range clamps to last copy.
        let off99 = tiled_offsets_with_index(&job, layout, trim_frames, cond_rate, 99);
        assert_eq!(off99.selected_body_start, head + 4 * body);

        // tail_start doesn't depend on selected index.
        assert_eq!(off0.tail_start, head + 5 * body);
        assert_eq!(off4.tail_start, head + 5 * body);
    }

    #[test]
    fn pad_for_engine_tiles_short_non_looped_sample() {
        // Non-looped sample (all loop modes = None) shorter than min_samples
        // should now tile back-to-back instead of fade+silence padding.
        let channels = 1;
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        let min_samples = 5000; // roughly 5x the sample
        let loop_info = SampleLoopInfo {
            normal: SampleLoopRegion::none(),
            sustain: SampleLoopRegion::none(),
        };

        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        let layout = layout.expect("expected Repeated layout for short non-looped sample");
        let repeated = layout.expect_repeated();

        assert!(
            repeated.copies >= 3,
            "minimum of 3 copies for tiling: got {}",
            repeated.copies
        );
        assert_eq!(repeated.copy_frames, 1000);
        assert_eq!(padded.len(), repeated.copies * data.len());

        // Every copy should be byte-identical to the original.
        for k in 0..repeated.copies {
            let start = k * data.len();
            let end = start + data.len();
            assert_eq!(&padded[start..end], &data[..]);
        }
    }

    #[test]
    fn pad_for_engine_non_looped_long_enough_stays_untiled() {
        let channels = 1;
        let data: Vec<f64> = (0..6000).map(|i| (i as f64 * 0.1).sin()).collect();
        let min_samples = 5000;
        let loop_info = SampleLoopInfo {
            normal: SampleLoopRegion::none(),
            sustain: SampleLoopRegion::none(),
        };
        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        assert!(
            layout.is_none(),
            "already-long sample should have no layout"
        );
        assert_eq!(padded, data);
    }

    #[test]
    fn extract_repeated_channel_output_picks_cleanest_copy() {
        // Non-looped synthetic sample; tile it 5 times, corrupt one copy, and
        // verify the extractor picks a clean one.
        let total = 1000usize;
        let reference: Vec<f64> = (0..total).map(|i| (i as f64 * 0.1).sin()).collect();
        let copies = 5;
        let corrupted_idx = 2;
        let mut raw = Vec::with_capacity(copies * total);
        for k in 0..copies {
            if k == corrupted_idx {
                raw.extend((0..total).map(|i| if i % 2 == 0 { 0.8 } else { -0.8 }));
            } else {
                raw.extend_from_slice(&reference);
            }
        }

        let mut job = mono_job_for_best_copy_test(
            reference.clone(),
            SampleLoopInfo {
                normal: SampleLoopRegion::none(),
                sustain: SampleLoopRegion::none(),
            },
            copies,
        );
        // Override layout to Repeated.
        job.engine_input_layout = Some(EngineInputLayout::Repeated(RepeatedLayout {
            copies,
            copy_frames: total,
        }));
        job.looped = false;
        job.original_length_48k_frames = total as i64;

        let extracted = extract_repeated_channel_output(
            &raw,
            1,
            &job,
            RepeatedLayout {
                copies,
                copy_frames: total,
            },
            total,
            48_000,
            0,
        );
        assert_eq!(extracted.len(), total);
        // The extracted copy should closely match the reference, except for
        // the fade-in window applied at the boundary when copy_index > 0.
        // Compare after the fade window.
        let skip = REPEATED_COPY_FADE_FRAMES;
        let mean_abs_diff: f64 = extracted
            .iter()
            .zip(reference.iter())
            .skip(skip)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / (total - skip) as f64;
        assert!(
            mean_abs_diff < 0.01,
            "extracted (post-fade) should match reference (got {mean_abs_diff})"
        );
    }

    #[test]
    fn round_up_to_four_k_plus_one_snaps_to_pingpong_forward_middle_grid() {
        assert_eq!(round_up_to_four_k_plus_one(0), 5);
        assert_eq!(round_up_to_four_k_plus_one(3), 5);
        assert_eq!(round_up_to_four_k_plus_one(5), 5);
        assert_eq!(round_up_to_four_k_plus_one(6), 9);
        assert_eq!(round_up_to_four_k_plus_one(7), 9);
        assert_eq!(round_up_to_four_k_plus_one(8), 9);
        assert_eq!(round_up_to_four_k_plus_one(9), 9);
        assert_eq!(round_up_to_four_k_plus_one(10), 13);
        for n in 0..200 {
            let m = round_up_to_four_k_plus_one(n);
            assert!(m >= 5);
            assert!(m >= n);
            assert_eq!(
                (m / 2) % 2,
                0,
                "body_copies={m} would place PingPong middle on a backward copy",
            );
        }
    }

    #[test]
    fn pad_for_engine_builds_alternating_fbfbf_layout_for_pingpong_loop() {
        let channels = 1usize;
        let head_frames = 8usize;
        let body_frames = 12usize;
        let tail_frames = 6usize;
        // Use a monotonic ramp for the body so forward vs backward copies are
        // distinguishable by content.
        let mut data = Vec::with_capacity(head_frames + body_frames + tail_frames);
        data.resize(head_frames, 0.0);
        for i in 0..body_frames {
            data.push(i as f64);
        }
        data.resize(head_frames + body_frames + tail_frames, -1.0);

        let loop_info =
            SampleLoopInfo::ping_pong(head_frames as i64, (head_frames + body_frames) as i64);
        // Force layered padding.
        let min_samples = (head_frames + 5 * body_frames + tail_frames) * 2;
        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        let layout = layout
            .expect("PingPong loop with padding must emit layered layout")
            .expect_single();

        // PingPong body_copies must be on the 4k+1 grid and ≥ 5.
        assert!(layout.body_copies >= 5);
        assert_eq!(layout.body_copies % 4, 1);

        // Layout: [head][F, B, F, B, F, …, F][tail]. Check the first three
        // body copies and the middle copy to confirm alternation.
        let forward: Vec<f64> = (0..body_frames).map(|i| i as f64).collect();
        let backward: Vec<f64> = (0..body_frames).rev().map(|i| i as f64).collect();

        let body_slice = |k: usize| -> &[f64] {
            let start = head_frames + k * body_frames;
            &padded[start..start + body_frames]
        };
        assert_eq!(
            body_slice(0),
            forward.as_slice(),
            "copy 0 should be forward"
        );
        assert_eq!(
            body_slice(1),
            backward.as_slice(),
            "copy 1 should be backward"
        );
        assert_eq!(
            body_slice(2),
            forward.as_slice(),
            "copy 2 should be forward"
        );

        // Middle copy (index body_copies/2) must be forward so extraction
        // yields a forward loop body.
        let middle_idx = layout.body_copies / 2;
        assert_eq!(
            body_slice(middle_idx),
            forward.as_slice(),
            "middle copy ({middle_idx}) must be forward"
        );

        // Tail appended verbatim at the end.
        let tail_start = head_frames + layout.body_copies * body_frames;
        assert!(
            padded[tail_start..tail_start + tail_frames]
                .iter()
                .all(|v| *v == -1.0)
        );
    }

    #[test]
    fn tiled_extraction_pulls_forward_middle_from_pingpong_layered_output() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");

        let rate = 48_000u32;
        let channels = 1usize;
        let head_frames = 16usize;
        let body_frames = 24usize;
        let tail_frames = 16usize;
        let total_frames = head_frames + body_frames + tail_frames;

        // Distinguishable regions: head=0.7, forward-body-ramp, tail=-0.9.
        let mut data = Vec::with_capacity(total_frames);
        data.resize(head_frames, 0.7);
        for i in 0..body_frames {
            data.push(i as f64 / body_frames as f64);
        }
        data.resize(total_frames, -0.9);

        let loop_info =
            SampleLoopInfo::ping_pong(head_frames as i64, (head_frames + body_frames) as i64);
        let min_samples = total_frames * 3;
        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        let layout = layout
            .expect("PingPong loop with padding must emit layered layout")
            .expect_single();

        // Identity engine: write the padded buffer back as the engine output.
        write_wav(&output_wav, &padded, rate, channels as u16).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "TiledPP".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total_frames as i64,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: total_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Single(layout)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result = extract_result(&job, &output_wav, "AudioSR")
            .expect("PingPong tiled extract should succeed");

        assert_eq!(result.length_frames, total_frames as i64);

        // Head and tail centers land on verbatim source values.
        assert_eq!(result.data[head_frames / 2], 0.7, "head center");
        let tail_center = head_frames + body_frames + tail_frames / 2;
        assert_eq!(result.data[tail_center], -0.9, "tail center");

        // Middle body should read as a forward ramp, not reversed.
        let body_center_frame = head_frames + body_frames / 2;
        let expected = (body_frames / 2) as f64 / body_frames as f64;
        assert!(
            (result.data[body_center_frame] - expected).abs() < 1e-9,
            "middle-body center at frame {body_center_frame}: expected {expected}, got {}",
            result.data[body_center_frame],
        );
        // And the body slope is positive (forward direction).
        let early = result.data[head_frames + 2];
        let late = result.data[head_frames + body_frames - 3];
        assert!(
            late > early,
            "middle-body should be increasing (forward); got early={early} late={late}",
        );
    }

    #[test]
    fn tiled_extraction_reconstructs_already_long_pingpong_loop_from_forward_middle() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");

        let rate = 48_000u32;
        let channels = 1usize;
        let head_frames = 24usize;
        let body_frames = 18usize;
        let tail_frames = 20usize;
        let total_frames = head_frames + body_frames + tail_frames;

        let mut data = Vec::with_capacity(total_frames);
        data.resize(head_frames, 0.7);
        for i in 0..body_frames {
            data.push(i as f64 / body_frames as f64);
        }
        data.resize(total_frames, -0.8);

        let loop_info =
            SampleLoopInfo::ping_pong(head_frames as i64, (head_frames + body_frames) as i64);
        let (padded, layout) = pad_for_engine(&data, channels, total_frames / 2, loop_info);
        let layout = layout
            .expect("already-long ping-pong loop should still be tiled")
            .expect_single();
        assert_eq!(layout.body_copies, 5);

        write_wav(&output_wav, &padded, rate, channels as u16).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "TiledLongPP".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total_frames as i64,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: total_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Single(layout)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result = extract_result(&job, &output_wav, "AudioSR")
            .expect("ping-pong tiled extract should succeed");
        assert_eq!(result.length_frames, total_frames as i64);
        assert_eq!(result.data[head_frames / 2], 0.7, "head center");
        assert_eq!(
            result.data[head_frames + body_frames + tail_frames / 2],
            -0.8,
            "tail center",
        );
        let early = result.data[head_frames + 2];
        let late = result.data[head_frames + body_frames - 3];
        assert!(late > early, "middle body should remain forward");
    }

    #[test]
    fn tiled_extraction_handles_sample_without_post_loop_tail() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");

        let rate = 48_000u32;
        let channels = 1usize;
        let head_frames = 24usize;
        let body_frames = 40usize;
        let total_frames = head_frames + body_frames;
        let (head_val, body_val) = (0.2, -0.4);

        let mut data = Vec::with_capacity(total_frames);
        data.resize(head_frames, head_val);
        data.resize(total_frames, body_val);

        // Loop ends exactly at sample end — no tail data.
        let loop_info = SampleLoopInfo::forward(head_frames as i64, total_frames as i64);
        let min_samples = total_frames * 4;
        let (padded, layout) = pad_for_engine(&data, channels, min_samples, loop_info);
        let layout = layout
            .expect("Forward loop with padding must emit layered layout")
            .expect_single();

        // Padded buffer is just [head][N×body] — no tail appended.
        assert_eq!(padded.len(), head_frames + layout.body_copies * body_frames);

        write_wav(&output_wav, &padded, rate, channels as u16).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "TiledNoTail".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total_frames as i64,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: total_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Single(layout)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result =
            extract_result(&job, &output_wav, "AudioSR").expect("tiled extract should succeed");
        assert_eq!(result.length_frames, total_frames as i64);
        assert_eq!(result.data[head_frames / 2], head_val);
        assert_eq!(result.data[head_frames + body_frames / 2], body_val);
    }

    #[test]
    fn extract_result_sustain_layout_reconstructs_middle_body_and_tail() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");

        let rate = 48_000u32;
        let channels = 1usize;
        let head_frames = 24usize;
        let body_frames = 18usize;
        let tail_frames = 20usize;
        let total_frames = head_frames + body_frames + tail_frames;
        let (head_val, body_val, tail_val) = (0.2, 0.6, -0.3);

        let mut data = Vec::with_capacity(total_frames);
        data.resize(head_frames, head_val);
        data.resize(head_frames + body_frames, body_val);
        data.resize(total_frames, tail_val);

        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::none(),
            SampleLoopRegion::forward(head_frames as i64, (head_frames + body_frames) as i64),
        );
        let (padded, layout) = pad_for_engine(&data, channels, total_frames / 2, loop_info);
        let layout = layout
            .expect("sustain-only loop should be tiled")
            .expect_single();
        assert_eq!(layout.loop_source, TiledLoopSource::Sustain);

        write_wav(&output_wav, &padded, rate, channels as u16).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "Sustain".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total_frames as i64,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_hold_input.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: total_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Single(layout)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let extracted =
            extract_result(&job, &output_wav, "AudioSR").expect("sustain extract should work");

        assert_eq!(extracted.length_frames, total_frames as i64);
        assert_eq!(extracted.data[head_frames / 2], head_val);
        assert_eq!(extracted.data[head_frames + body_frames / 2], body_val);
        assert_eq!(
            extracted.data[head_frames + body_frames + tail_frames / 2],
            tail_val
        );
    }

    // Test removed: extract_result_lavasr_sustain_layout_resamples_hold_and_release_outputs
    // LavaSR-specific sustain layout and extract_sustain_result were removed in the
    // pipeline simplification.

    #[test]
    fn extract_result_low_rate_one_shot_trims_to_scaled_length() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("sample_0.wav");
        let source_rate = 8_000u32;
        let source_frames = 128i64;
        let scaled_frames = scaled_frame_count(source_frames as usize, source_rate, 48_000);

        // Engine output is larger than needed (e.g. from padding).
        let extra_frames = 64;
        let mut engine_output = Vec::with_capacity(scaled_frames + extra_frames);
        for i in 0..(scaled_frames + extra_frames) {
            engine_output.push(0.1 * i as f64);
        }
        write_wav(&output_wav, &engine_output, 48_000, 1).expect("should write engine output");

        let job = SampleJob {
            index: 0,
            name: "LowRateOneShot".into(),
            original_data: vec![0.0; source_frames as usize],
            rate: source_rate as i32,
            output_sample_rate_hz: 48_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: source_frames,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("sample_0_input.wav"),
                scaled_frames as i64,
            ),
            conditioning_rate_hz: 48_000,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: vec![0.0; scaled_frames],
            original_length_48k_frames: scaled_frames as i64,
            engine_input_layout: None,
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let extracted =
            extract_result(&job, &output_wav, "AudioSR").expect("one-shot extract should work");

        // extract_channel_result trims to original_length_48k_frames
        assert_eq!(extracted.length_frames, scaled_frames as i64);
        assert_eq!(extracted.data.len(), scaled_frames);
        assert_eq!(
            &extracted.data[..],
            &engine_output[..scaled_frames],
            "trimmed result must equal first scaled_frames of engine output",
        );
    }

    #[test]
    fn loop_seam_fix_search_and_fallback_reduce_forward_wrap_score() {
        let channels = 1usize;
        let loop_start = 100usize;
        let loop_end = 320usize;
        let frames = 360usize;
        let mut data = vec![0.0f64; frames * channels];
        for i in 0..frames {
            data[i] = 0.3 * (i as f64 * 0.07).sin();
        }

        // Create a bad seam at the declared loop start.
        data[loop_start] = 0.8;
        data[loop_start + 1] = 0.82;
        data[loop_end - 2] = -0.21;
        data[loop_end - 1] = -0.2;

        // Inject a better-matching nearby phase candidate.
        let candidate = 112usize;
        data[candidate] = -0.2;
        data[candidate + 1] = -0.19;

        let score_before = seam_score_at_forward_wrap(
            &data,
            channels,
            loop_start,
            loop_end,
            DEFAULT_LOOP_SEAM_FIX_OPTIONS.amplitude_weight,
            DEFAULT_LOOP_SEAM_FIX_OPTIONS.slope_weight,
        );
        let _ = crossfade_loop_boundary(&mut data, channels, loop_start as i64, loop_end as i64);
        let score_after = seam_score_at_forward_wrap(
            &data,
            channels,
            loop_start,
            loop_end,
            DEFAULT_LOOP_SEAM_FIX_OPTIONS.amplitude_weight,
            DEFAULT_LOOP_SEAM_FIX_OPTIONS.slope_weight,
        );

        assert!(
            score_after < score_before,
            "loop seam score should improve: before={score_before}, after={score_after}"
        );
        assert_eq!(
            data[loop_start],
            data[loop_end - 1],
            "loop endpoint should be force-matched"
        );
    }

    #[test]
    fn crossfade_returns_max_inner_jump_when_endpoint_force_creates_cliff() {
        // Mirror Twilight.umx sample #22: loop-start sample sits near -0.05, but
        // a window's worth of frames into the loop the signal is near -0.33. The
        // crossfade copies head[window-1] into data[end-1], then endpoint-force
        // overwrites it with data[loop_start]. Result: data[end-2] near -0.33,
        // data[end-1] near -0.05 — a big cliff.
        let channels = 1usize;
        let loop_start = 100usize;
        let loop_end = 320usize;
        let frames = 360usize;
        let mut data = vec![-0.33f64; frames * channels];
        data[loop_start] = -0.05;

        let jump = crossfade_loop_boundary(&mut data, channels, loop_start as i64, loop_end as i64);
        assert!(
            jump >= 0.2,
            "crossfade should report the endpoint-force cliff as max_inner_jump (got {jump})"
        );
    }

    #[test]
    fn boundary_repair_guard_reverts_non_improving_edits() {
        let channels = 1usize;
        let loop_region = SampleLoopRegion::forward(16, 96);
        let mut data: Vec<f64> = (0..128).map(|i| 0.4 * (i as f64 * 0.05).sin()).collect();
        let original = data.clone();

        let accepted = apply_boundary_repair_with_guard(
            &mut data,
            channels,
            loop_region,
            "test_revert",
            |buf| {
                let ls = loop_region.start_frames as usize;
                let le = loop_region.end_frames as usize;
                buf[ls] = 1.0;
                buf[le - 1] = -1.0;
                buf[ls + 1] = 1.0;
                buf[le - 2] = -1.0;
            },
        );

        assert!(
            !accepted,
            "guard should reject edits that do not improve seam metrics"
        );
        assert_eq!(data, original, "rejected edits must be rolled back");
    }

    #[test]
    fn boundary_repair_guard_accepts_improving_edits() {
        let channels = 1usize;
        let loop_region = SampleLoopRegion::forward(20, 120);
        let mut data: Vec<f64> = (0..160).map(|i| 0.3 * (i as f64 * 0.04).sin()).collect();

        // Force a bad seam.
        let ls = loop_region.start_frames as usize;
        let le = loop_region.end_frames as usize;
        data[ls] = -0.9;
        data[ls + 1] = -0.75;
        data[le - 1] = 0.9;
        data[le - 2] = 0.75;
        let pre_metrics = compute_loop_boundary_metrics(&data, channels, ls, le);

        let accepted = apply_boundary_repair_with_guard(
            &mut data,
            channels,
            loop_region,
            "test_accept",
            |buf| {
                buf[le - 1] = buf[ls];
                buf[le - 2] = buf[ls + 1];
            },
        );
        let post_metrics = compute_loop_boundary_metrics(&data, channels, ls, le);

        assert!(accepted, "guard should keep edits that improve seam safely");
        assert!(
            post_metrics.seam_score < pre_metrics.seam_score,
            "seam score should improve when edit is accepted"
        );
    }

    #[test]
    fn boundary_repair_guard_skips_good_enough_seam() {
        let channels = 1usize;
        let loop_region = SampleLoopRegion::forward(20, 120);
        let mut data: Vec<f64> = (0..160).map(|i| 0.28 * (i as f64 * 0.03).sin()).collect();
        let original = data.clone();
        let ls = loop_region.start_frames as usize;
        let le = loop_region.end_frames as usize;

        // Keep seam below the configured good-enough threshold.
        data[le - 1] = data[ls] + 0.03;
        data[le - 2] = data[ls + 1] + 0.02;
        let pre = compute_loop_boundary_metrics(&data, channels, ls, le);
        assert!(
            pre.seam_score <= LOOP_REPAIR_GOOD_ENOUGH_SEAM_SCORE,
            "test setup must stay below skip threshold (got {})",
            pre.seam_score
        );

        let mut attempted = false;
        let accepted = apply_boundary_repair_with_guard(
            &mut data,
            channels,
            loop_region,
            "test_skip",
            |buf| {
                attempted = true;
                buf[le - 1] = -1.0;
                buf[le - 2] = -1.0;
            },
        );

        assert!(!accepted, "good-enough seam should skip repair");
        assert!(!attempted, "repair closure should not run when skipped");
        // unchanged vs pre-skip state
        assert_eq!(data, {
            let mut expected = original;
            expected[le - 1] = expected[ls] + 0.03;
            expected[le - 2] = expected[ls + 1] + 0.02;
            expected
        });
    }

    #[test]
    fn assemble_sample_candidate_reinterleaves_stereo_channels_in_index_order() {
        let job = SampleJob {
            index: 0,
            name: "Stereo".into(),
            original_data: vec![0.0; 8],
            rate: 48_000,
            output_sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            source_length_frames: 4,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            conditioning_inputs: vec![
                PreparedChannelInput {
                    channel_index: 0,
                    channel_name: "left".into(),
                    input_path: PathBuf::from("sample_0_L.wav"),
                    input_length_frames: 4,
                },
                PreparedChannelInput {
                    channel_index: 1,
                    channel_name: "right".into(),
                    input_path: PathBuf::from("sample_0_R.wav"),
                    input_length_frames: 4,
                },
            ],
            conditioning_rate_hz: 48_000,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: vec![0.0; 8],
            original_length_48k_frames: 4,
            engine_input_layout: None,
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let assembled = assemble_sample_candidate(
            &job,
            "AudioSR",
            &[
                ChannelResult {
                    channel_index: 1,
                    data: vec![10.0, 20.0, 30.0],
                    length_frames: 3,
                },
                ChannelResult {
                    channel_index: 0,
                    data: vec![1.0, 2.0, 3.0, 4.0],
                    length_frames: 4,
                },
            ],
        )
        .expect("stereo candidate should assemble");

        assert_eq!(assembled.channels, 2);
        assert_eq!(assembled.length_frames, 3);
        assert_eq!(assembled.data, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn normalize_conditioning_input_removes_mono_dc_bias_and_targets_half_scale_peak() {
        let mut conditioning = vec![0.25, 0.5, 0.75, 1.0];

        normalize_conditioning_input(&mut conditioning, 1);

        let mean = conditioning
            .iter()
            .map(|&sample| sample as f64)
            .sum::<f64>()
            / conditioning.len() as f64;
        assert!(mean.abs() < 1.0e-6, "mean should be near zero, got {mean}");
        assert!((simd::peak_abs_f64(&conditioning) - CONDITIONING_TARGET_PEAK).abs() < 1.0e-6);
    }

    #[test]
    fn normalize_conditioning_input_normalizes_stereo_channels_independently() {
        let mut conditioning = vec![0.25, -0.75, 0.75, -0.25, 1.25, 0.25, 1.75, 0.75];

        normalize_conditioning_input(&mut conditioning, 2);

        let (left, right) = simd::deinterleave_stereo_f64(&conditioning);
        let left_mean = left.iter().map(|&sample| sample as f64).sum::<f64>() / left.len() as f64;
        let right_mean =
            right.iter().map(|&sample| sample as f64).sum::<f64>() / right.len() as f64;
        assert!(
            left_mean.abs() < 1.0e-6,
            "left mean should be near zero, got {left_mean}"
        );
        assert!(
            right_mean.abs() < 1.0e-6,
            "right mean should be near zero, got {right_mean}"
        );
        assert!((simd::peak_abs_f64(&left) - CONDITIONING_TARGET_PEAK).abs() < 1.0e-6);
        assert!((simd::peak_abs_f64(&right) - CONDITIONING_TARGET_PEAK).abs() < 1.0e-6);
    }

    #[test]
    fn normalize_conditioning_input_leaves_silence_unchanged() {
        let mut conditioning = vec![0.0; 8];

        normalize_conditioning_input(&mut conditioning, 2);

        assert_eq!(conditioning, vec![0.0; 8]);
    }

    #[test]
    fn spectral_correlation_uses_original_rate_not_conditioning_rate() {
        let sample_rate = 48_000.0f64;
        let frames = 4096usize;
        let reference: Vec<f64> = (0..frames)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.6 * (2.0 * std::f64::consts::PI * 2_000.0 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 12_000.0 * t).sin()
            })
            .collect();
        let candidate: Vec<f64> = (0..frames)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.6 * (2.0 * std::f64::consts::PI * 2_000.0 * t).sin()
            })
            .collect();

        let narrow = crate::engine::spectral_correlation(&reference, &candidate, 1, 8_000);
        let wide = crate::engine::spectral_correlation(&reference, &candidate, 1, 48_000);

        assert!(
            narrow > wide,
            "original-rate-limited scoring should ignore the 12 kHz mismatch"
        );
        assert!(narrow > 0.99, "narrow-band score should stay near-perfect");
        assert!(
            wide < 0.95,
            "wide-band score should penalize the missing 12 kHz content"
        );
    }

    #[test]
    fn repair_forward_loop_tail_reduces_wrap_gap_without_touching_attack_prefix() {
        let mut repaired = vec![0.05, 0.2, 0.85, -0.1, 0.35, -0.45];
        let original = repaired.clone();
        let before = forward_loop_seam_discontinuity(&repaired, 1, 2);
        let repaired_frames = repaired.len() as i64;

        repair_forward_loop_tail_in_place(&mut repaired, 1, 2, repaired_frames);

        let after = forward_loop_seam_discontinuity(&repaired, 1, 2);
        assert!(
            after < before,
            "forward-loop repair should reduce the wrap gap (before={before}, after={after})",
        );
        assert_eq!(
            &repaired[..2],
            &original[..2],
            "forward-loop repair should leave the attack prefix untouched",
        );
    }

    #[test]
    fn sustain_reference_repairs_loop_before_attack_and_release_alignment() {
        let original = vec![0.05, 0.1, 0.4, 0.6, 0.8, 0.65, 0.45, 0.25];
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(5, 8),
            SampleLoopRegion::forward(2, 5),
        );
        let loop_prep = LoopPrepPlan::from_sample(original.len() as i64, loop_info);
        let saved_original = sample_prefix(&original, 1, loop_prep.saved_length_frames);
        let hold_original = sample_prefix(saved_original, 1, loop_prep.sustain_loop.end_frames);
        let raw_hold = build_canonical_segment_reference(
            hold_original,
            16_000,
            1,
            loop_prep.sustain_loop,
            off_settings(),
        )
        .expect("raw hold reference should build");
        let raw_release = build_canonical_segment_reference(
            saved_original,
            16_000,
            1,
            loop_prep.post_keyoff_loop(),
            off_settings(),
        )
        .expect("raw release reference should build");
        let reference_parts = build_canonical_reference_parts_with_loop_prep(
            &original,
            16_000,
            1,
            loop_prep,
            off_settings(),
        )
        .expect("repaired sustain reference should build");
        let repaired_hold = reference_parts
            .hold_reference
            .expect("sustain repair should keep a repaired hold segment");
        let repaired_release = reference_parts
            .release_reference
            .expect("sustain repair should keep a repaired release sample");

        assert_eq!(
            repaired_release, reference_parts.final_reference,
            "the sustain-aligned release sample should be the canonical one-copy reference",
        );
        assert!(
            forward_loop_seam_discontinuity(
                &repaired_hold,
                1,
                loop_prep.sustain_loop.start_frames as usize,
            ) < forward_loop_seam_discontinuity(
                &raw_hold,
                1,
                loop_prep.sustain_loop.start_frames as usize,
            ),
            "the sustain loop body should be repaired before it anchors later stages",
        );
        assert!(
            adjacent_jump_at(
                &repaired_hold,
                1,
                loop_prep.sustain_loop.start_frames as usize
            ) < adjacent_jump_at(&raw_hold, 1, loop_prep.sustain_loop.start_frames as usize),
            "the attack tail should enter the repaired sustain loop more smoothly",
        );
        let sustain_tail = repaired_hold[(loop_prep.sustain_loop.end_frames - 1) as usize];
        assert!(
            (repaired_release[loop_prep.sustain_loop.end_frames as usize] - sustain_tail).abs()
                < (raw_release[loop_prep.sustain_loop.end_frames as usize] - sustain_tail).abs(),
            "the release head should be anchored to the repaired sustain tail",
        );

        let cancel_flag = AtomicBool::new(false);
        let prepared = prepare_sample_data(
            &original,
            16_000,
            1,
            loop_prep,
            off_settings(),
            0.01,
            &cancel_flag,
        )
        .expect("prepared sustain sample should build");
        assert_eq!(
            prepared.reference_data, reference_parts.final_reference,
            "prepared reference data should use the repaired sustain-aligned one-copy sample",
        );
        assert_eq!(
            prepared
                .release_reference_data
                .as_ref()
                .expect("release reference"),
            &reference_parts.final_reference,
            "release model input should be built from the sustain-aligned one-copy sample",
        );
    }

    #[test]
    fn sustain_post_keyoff_loop_repair_uses_stitched_sample_as_wrap_target() {
        let original = vec![0.1, 0.2, 0.75, 0.35, -0.4, 0.5, -0.1, 0.7];
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(3, 8),
            SampleLoopRegion::forward(2, 5),
        );
        let loop_prep = LoopPrepPlan::from_sample(original.len() as i64, loop_info);
        let saved_original = sample_prefix(&original, 1, loop_prep.saved_length_frames);
        let hold_original = sample_prefix(saved_original, 1, loop_prep.sustain_loop.end_frames);

        let mut hold_reference = build_canonical_segment_reference(
            hold_original,
            16_000,
            1,
            loop_prep.sustain_loop,
            off_settings(),
        )
        .expect("hold reference should build");
        repair_loop_body_in_place(&mut hold_reference, 1, loop_prep.sustain_loop);
        repair_attack_tail_to_loop_head_in_place(
            &mut hold_reference,
            1,
            loop_prep.sustain_loop.start_frames,
        );

        let mut release_reference = build_canonical_segment_reference(
            saved_original,
            16_000,
            1,
            loop_prep.post_keyoff_loop(),
            off_settings(),
        )
        .expect("release reference should build");
        let hold_tail_anchor =
            frame_values(&hold_reference, 1, loop_prep.sustain_loop.end_frames - 1);
        repair_release_head_to_loop_tail_in_place(
            &mut release_reference,
            &hold_tail_anchor,
            1,
            loop_prep.sustain_loop.end_frames,
        );
        let sustain_window = loop_prep
            .sustain_loop
            .end_frames
            .min(loop_prep.saved_length_frames - loop_prep.sustain_loop.end_frames)
            .max(0);
        let stitched_without_post_keyoff_repair = stitch_sustain_references(
            &hold_reference,
            &release_reference,
            1,
            loop_prep.sustain_loop.end_frames,
            sustain_window,
        );

        let final_reference = build_canonical_reference_parts_with_loop_prep(
            &original,
            16_000,
            1,
            loop_prep,
            off_settings(),
        )
        .expect("full sustain reference should build")
        .final_reference;
        assert!(
            forward_loop_seam_discontinuity(
                &final_reference,
                1,
                loop_prep.normal_loop.start_frames as usize,
            ) < forward_loop_seam_discontinuity(
                &stitched_without_post_keyoff_repair,
                1,
                loop_prep.normal_loop.start_frames as usize,
            ),
            "post-keyoff loop repair should use the stitched sustain-derived prefix as its wrap target",
        );
    }

    #[test]
    fn ping_pong_sustain_repair_aligns_entry_and_release_after_turnarounds() {
        let sustain_loop = SampleLoopRegion::ping_pong(2, 5);
        let sustain_end = sustain_loop.end_frames;
        let raw_hold = vec![0.1, -0.55, 0.85, -0.2, 0.75];
        let mut turnaround_repaired = raw_hold.clone();

        let raw_start_turnaround =
            frame_gap_between(&raw_hold, 1, sustain_loop.start_frames as usize, 3);
        let raw_end_turnaround = frame_gap_between(&raw_hold, 1, 4, 3);
        repair_loop_body_in_place(&mut turnaround_repaired, 1, sustain_loop);

        let mut fully_repaired_hold = turnaround_repaired.clone();
        repair_attack_tail_to_loop_head_in_place(
            &mut fully_repaired_hold,
            1,
            sustain_loop.start_frames,
        );
        let release = vec![0.05, -0.1, 0.2, -0.25, 0.3, -0.8, -0.2, 0.1];
        let sustain_window = sustain_end.min(release.len() as i64 - sustain_end).max(0);
        let baseline_stitched = stitch_sustain_references(
            &fully_repaired_hold,
            &release,
            1,
            sustain_end,
            sustain_window,
        );
        let hold_tail_anchor = frame_values(&fully_repaired_hold, 1, sustain_end - 1);
        let mut repaired_release = release.clone();
        repair_release_head_to_loop_tail_in_place(
            &mut repaired_release,
            &hold_tail_anchor,
            1,
            sustain_end,
        );
        let repaired_stitched = stitch_sustain_references(
            &fully_repaired_hold,
            &repaired_release,
            1,
            sustain_end,
            sustain_window,
        );

        assert!(
            frame_gap_between(
                &turnaround_repaired,
                1,
                sustain_loop.start_frames as usize,
                3,
            ) < raw_start_turnaround,
            "ping-pong repair should smooth the start turnaround before entry alignment",
        );
        assert!(
            frame_gap_between(&turnaround_repaired, 1, 4, 3) < raw_end_turnaround,
            "ping-pong repair should smooth the end turnaround before release alignment",
        );
        assert!(
            adjacent_jump_at(&fully_repaired_hold, 1, sustain_loop.start_frames as usize)
                < adjacent_jump_at(&turnaround_repaired, 1, sustain_loop.start_frames as usize),
            "attack-tail alignment should run after the turnaround repair",
        );
        assert!(
            adjacent_jump_at(&repaired_stitched, 1, sustain_end as usize)
                < adjacent_jump_at(&baseline_stitched, 1, sustain_end as usize),
            "release-head alignment should run after the repaired hold loop becomes the anchor",
        );
    }

    #[test]
    fn skav_starship_reference_and_extract_reduce_loop_seam() {
        let data = std::fs::read("mods/2ND_SKAV.S3M").expect("fixture should load");
        let mut module = Module::from_memory(&data).expect("fixture module should load");
        let sample = read_raw_samples(&mut module)
            .into_iter()
            .find(|sample| sample.index == 8)
            .expect("2ND_SKAV sample 9 should be readable");
        let channels = sample.channels as usize;
        let loop_prep = LoopPrepPlan::from_sample(sample.source_length_frames, sample.loop_info);
        assert!(
            loop_prep.uses_boundary_loop(),
            "2ND_SKAV sample 9 should use the boundary-loop path",
        );

        let saved_original = sample_prefix(&sample.data, channels, loop_prep.saved_length_frames);
        let original_gap = forward_loop_seam_discontinuity(
            saved_original,
            channels,
            loop_prep.normal_loop.start_frames as usize,
        );
        let repaired_reference = build_quinlight_reference_48k_with_loop_info(
            &sample.data,
            sample.rate as u32,
            channels,
            sample.loop_info,
            off_settings(),
        )
        .expect("reference build should succeed");
        let scaled_loop_start = scaled_frame_count(
            loop_prep.normal_loop.start_frames as usize,
            sample.rate as u32,
            48_000,
        );
        let repaired_gap =
            forward_loop_seam_discontinuity(&repaired_reference, channels, scaled_loop_start);
        assert!(
            repaired_gap < original_gap,
            "pre-AI repair should reduce the forward-loop seam for 2ND_SKAV sample 9 (original_gap={original_gap}, repaired_gap={repaired_gap})",
        );

        let filtered_reference_parts = build_canonical_reference_parts_with_loop_prep(
            &sample.data,
            sample.rate as u32,
            channels,
            loop_prep,
            off_settings(),
        )
        .expect("filtered reference parts should build");
        let filtered_reference = filtered_reference_parts.final_reference;
        let filtered_engine_output = resample_segment_reference(
            &filtered_reference,
            sample.rate as u32,
            48_000,
            channels,
            loop_prep.normal_loop,
            None,
        )
        .expect("filtered engine output should build");
        let guided_engine_output = resample_reference_with_loop_prep(
            &filtered_reference,
            None,
            None,
            sample.rate as u32,
            48_000,
            channels,
            loop_prep,
            None,
            Some(sample.index),
        )
        .expect("guided engine output should build");
        let filtered_gap =
            forward_loop_seam_discontinuity(&filtered_engine_output, channels, scaled_loop_start);
        let guided_gap =
            forward_loop_seam_discontinuity(&guided_engine_output, channels, scaled_loop_start);
        let scaled_loop_end = scaled_frame_count(
            loop_prep.normal_loop.end_frames as usize,
            sample.rate as u32,
            48_000,
        );
        let filtered_inner_jump = max_adjacent_jump_in_range(
            &filtered_engine_output,
            channels,
            scaled_loop_start,
            scaled_loop_end,
        );
        let guided_inner_jump = max_adjacent_jump_in_range(
            &guided_engine_output,
            channels,
            scaled_loop_start,
            scaled_loop_end,
        );
        assert!(
            guided_gap <= filtered_gap + 0.01,
            "source-guided SINC should keep the repaired seam at least as smooth \
             (filtered={filtered_gap}, guided={guided_gap})",
        );
        // The phase-anchored spectral blend targets seam continuity, not
        // inner-loop HF clipping. When the candidate is the sinc-resampled
        // source (source ≈ candidate), the blend is effectively a pass-through
        // so the inner jump stays approximately the same. The rotor-correct
        // blend (`rotor::polar_lerp`) computes phase via the Cartesian-blend
        // arg + magnitude via arithmetic lerp, which introduces sub-percent
        // numerical drift relative to the older Cartesian linear blend even
        // when src ≈ cand. Allow ~5% tolerance to absorb that drift while
        // still catching a real regression.
        let inner_jump_tol = (filtered_inner_jump * 0.05).max(1e-3);
        assert!(
            guided_inner_jump <= filtered_inner_jump + inner_jump_tol,
            "source-guided SINC should not noticeably increase the in-loop spike for \
             2ND_SKAV sample 9 (filtered={filtered_inner_jump}, guided={guided_inner_jump}, \
             tol={inner_jump_tol})",
        );
    }

    #[test]
    fn cleanup_settings_change_cache_hash() {
        let original: Vec<f64> = (0..2048)
            .map(|i| {
                let t = i as f64 / 12_000.0;
                0.6 * (2.0 * std::f64::consts::PI * 180.0 * t).sin()
                    + 0.2
                    + if i == 97 { 1.0 } else { 0.0 }
            })
            .collect();

        let off_v1 = build_canonical_reference(
            &original,
            12_000,
            1,
            false,
            cleanup_settings(CleanupMode::Off, CleanupEngineVersion::V1),
        )
        .expect("off reference should build");
        let off_v21 = build_canonical_reference(
            &original,
            12_000,
            1,
            false,
            cleanup_settings(CleanupMode::Off, CleanupEngineVersion::V21),
        )
        .expect("off reference should build");

        let off_v1_hash = compute_sample_hash(
            &off_v1,
            12_000,
            1,
            false,
            cleanup_settings(CleanupMode::Off, CleanupEngineVersion::V1),
        );
        let off_v21_hash = compute_sample_hash(
            &off_v21,
            12_000,
            1,
            false,
            cleanup_settings(CleanupMode::Off, CleanupEngineVersion::V21),
        );
        assert_eq!(off_v1_hash, off_v21_hash);

        for mode in [
            CleanupMode::DeclickAr,
            CleanupMode::DeclickMedian,
            CleanupMode::Decrackle,
        ] {
            let v1 = cleanup_settings(mode, CleanupEngineVersion::V1);
            let v21 = cleanup_settings(mode, CleanupEngineVersion::V21);
            let v1_ref =
                build_canonical_reference(&original, 12_000, 1, false, v1).expect("v1 build");
            let v21_ref =
                build_canonical_reference(&original, 12_000, 1, false, v21).expect("v21 build");
            let v1_hash = compute_sample_hash(&v1_ref, 12_000, 1, false, v1);
            let v21_hash = compute_sample_hash(&v21_ref, 12_000, 1, false, v21);

            assert_ne!(off_v1_hash, v1_hash);
            assert_ne!(off_v1_hash, v21_hash);
            assert_ne!(v1_hash, v21_hash);
        }
    }

    #[test]
    fn forward_loop_cache_hash_changes_with_loop_boundaries() {
        let reference = vec![0.1, 0.2, 0.3, 0.4];
        let loop_a = LoopPrepPlan::from_sample(6, SampleLoopInfo::forward(2, 4));
        let loop_b = LoopPrepPlan::from_sample(6, SampleLoopInfo::forward(1, 4));

        let hash_a =
            compute_sample_hash_for_loop_prep(&reference, 16_000, 1, loop_a, off_settings());
        let hash_b =
            compute_sample_hash_for_loop_prep(&reference, 16_000, 1, loop_b, off_settings());

        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn looped_click_median_cleanup_keeps_seams_stable_for_both_engines() {
        let looped: Vec<f64> = (0..256)
            .map(|i| {
                let t = i as f64 / 8_000.0;
                0.6 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                    + 0.15 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
            })
            .collect();

        let off = build_quinlight_reference_48k(&looped, 8_000, 1, true, off_settings())
            .expect("off loop-aware reference should build");

        for engine_version in [CleanupEngineVersion::V1, CleanupEngineVersion::V21] {
            let click_median = build_quinlight_reference_48k(
                &looped,
                8_000,
                1,
                true,
                cleanup_settings(CleanupMode::DeclickMedian, engine_version),
            )
            .expect("click median loop-aware reference should build");

            assert!(
                seam_discontinuity(&click_median, 1) <= seam_discontinuity(&off, 1) + 0.01,
                "click median cleanup should not introduce a worse seam discontinuity than the legacy loop-aware reference",
            );
        }
    }

    fn impulsive_peak_fixture() -> Vec<f64> {
        let mut original = vec![0.0f64; 4096];
        for (i, sample) in original.iter_mut().enumerate() {
            let t = i as f64 / 16_000.0;
            *sample = 0.25 * (2.0 * std::f64::consts::PI * 220.0 * t).sin();
        }
        for sample in &mut original[1200..1204] {
            *sample = 1.0;
        }
        for sample in &mut original[2400..2404] {
            *sample = -1.0;
        }
        original
    }

    fn assert_cleanup_reduces_impulsive_peaks(
        mode: CleanupMode,
        engine_version: CleanupEngineVersion,
    ) {
        let original = impulsive_peak_fixture();
        let off = build_canonical_reference(&original, 16_000, 1, false, off_settings())
            .expect("off reference should build");
        let cleaned = build_canonical_reference(
            &original,
            16_000,
            1,
            false,
            cleanup_settings(mode, engine_version),
        )
        .expect("cleanup reference should build");

        let off_clipped = off.iter().filter(|sample| sample.abs() >= 0.99).count();
        let cleaned_clipped = cleaned.iter().filter(|sample| sample.abs() >= 0.99).count();

        assert!(cleaned_clipped < off_clipped);
    }

    #[test]
    fn declick_ar_cleanup_reduces_impulsive_peaks_for_both_engines() {
        for engine_version in [CleanupEngineVersion::V1, CleanupEngineVersion::V21] {
            assert_cleanup_reduces_impulsive_peaks(CleanupMode::DeclickAr, engine_version);
        }
    }

    #[test]
    fn declick_median_cleanup_reduces_impulsive_peaks_for_both_engines() {
        for engine_version in [CleanupEngineVersion::V1, CleanupEngineVersion::V21] {
            assert_cleanup_reduces_impulsive_peaks(CleanupMode::DeclickMedian, engine_version);
        }
    }

    #[test]
    fn decrackle_cleanup_reduces_transient_intensity_for_both_engines() {
        let original: Vec<f64> = (0..8192)
            .map(|i| {
                let t = i as f64 / 8_000.0;
                let crackle = if i % 173 == 0 { 0.9 } else { 0.0 };
                0.35 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() + crackle
            })
            .collect();

        let off = build_canonical_reference(&original, 8_000, 1, false, off_settings())
            .expect("off reference should build");

        for engine_version in [CleanupEngineVersion::V1, CleanupEngineVersion::V21] {
            let crackle = build_canonical_reference(
                &original,
                8_000,
                1,
                false,
                cleanup_settings(CleanupMode::Decrackle, engine_version),
            )
            .expect("crackle reference should build");

            assert!(
                transient_intensity(&crackle) < transient_intensity(&off),
                "crackle cleanup should reduce transient intensity",
            );
        }
    }

    #[test]
    fn cleanup_hash_version_bump_invalidates_prior_entries() {
        let original: Vec<f64> = (0..2048)
            .map(|i| {
                let t = i as f64 / 12_000.0;
                0.5 * (2.0 * std::f64::consts::PI * 210.0 * t).sin()
                    + if (512..516).contains(&i) { 1.0 } else { 0.0 }
            })
            .collect();

        for cleanup_settings in all_public_cleanup_settings() {
            let reference =
                build_canonical_reference(&original, 12_000, 1, false, cleanup_settings)
                    .expect("cleanup reference should build");
            let current = compute_sample_hash(&reference, 12_000, 1, false, cleanup_settings);
            let previous = compute_sample_hash_with_version(
                &reference,
                12_000,
                1,
                false,
                cleanup_settings.hash_tag(),
                PREVIOUS_SAMPLE_CACHE_HASH_VERSION,
            );
            assert_ne!(current, previous);
        }

        let off_reference = build_canonical_reference(&original, 12_000, 1, false, off_settings())
            .expect("off reference should build");
        let current = compute_sample_hash(&off_reference, 12_000, 1, false, off_settings());
        let previous = compute_sample_hash_with_version(
            &off_reference,
            12_000,
            1,
            false,
            off_settings().hash_tag(),
            PREVIOUS_SAMPLE_CACHE_HASH_VERSION,
        );
        assert_ne!(current, previous);
    }

    #[test]
    fn apply_sample_replacement_repatches_from_saved_xm_effect_snapshot() {
        let (mut module, saved_effects, replacement, expected_param) = xm_offset_apply_fixture();

        assert_eq!(saved_effects, vec![(0, 0, 0, 0x10)]);
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, expected_param));
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Fixture should begin in an already-scaled state",
        );

        apply_sample_replacement(
            &mut module,
            1,
            &replacement,
            replacement.len() as i64,
            1,
            48_000,
            16_000,
            &saved_effects,
        )
        .expect("Shared sample apply should succeed");

        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            expected_param,
            "Shared sample apply should restore original XM offsets before repatching",
        );
    }

    #[test]
    fn engine_batch_manifest_writes_rate_derived_bandwidth_metadata() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let job = sample_job(work_dir.path());
        let manifest_path = work_dir.path().join("manifest.json");
        let input = job
            .conditioning_inputs
            .first()
            .expect("mono job should have one conditioning input");
        write_engine_batch_manifest(
            &manifest_path,
            vec![engine_batch_item_for_input(&job, input)],
        )
        .expect("manifest should write");

        let manifest_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).expect("manifest should read"))
                .expect("manifest should parse");
        assert_eq!(manifest_json["version"], 1);
        assert_eq!(manifest_json["items"][0]["stem"], "sample_0_mono");
        assert_eq!(manifest_json["items"][0]["sample_index"], 0);
        assert_eq!(manifest_json["items"][0]["source_stem"], "sample_0");
        assert_eq!(
            manifest_json["items"][0]["conditioning_wav_path"],
            input.input_path.to_string_lossy().as_ref()
        );
        assert_eq!(manifest_json["items"][0]["original_rate_hz"], 16_000);
        assert_eq!(manifest_json["items"][0]["original_nyquist_hz"], 8_000.0);
        assert_eq!(manifest_json["items"][0]["conditioning_rate_hz"], 24_000);
        assert_eq!(
            manifest_json["items"][0]["conditioning_lowpass_hz"],
            7_200.0
        );
        assert_eq!(manifest_json["items"][0]["source_channels"], 1);
        assert_eq!(manifest_json["items"][0]["conditioning_channels"], 1);
        assert_eq!(manifest_json["items"][0]["channel_index"], 0);
        assert_eq!(manifest_json["items"][0]["channel_name"], "mono");
    }

    #[test]
    fn stereo_manifest_expands_to_left_and_right_channel_items() {
        let job = SampleJob {
            index: 0,
            name: "Stereo".into(),
            original_data: vec![0.0; 1024],
            rate: 16_000,
            output_sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            source_length_frames: 512,
            looped: false,
            loop_info: SampleLoopInfo::none(),
            conditioning_inputs: vec![
                PreparedChannelInput {
                    channel_index: 0,
                    channel_name: "left".into(),
                    input_path: PathBuf::from("sample_0_L.wav"),
                    input_length_frames: 1536,
                },
                PreparedChannelInput {
                    channel_index: 1,
                    channel_name: "right".into(),
                    input_path: PathBuf::from("sample_0_R.wav"),
                    input_length_frames: 1536,
                },
            ],
            conditioning_rate_hz: 48_000,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: vec![0.0; 1024],
            original_length_48k_frames: 512,
            engine_input_layout: None,
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };
        let items: Vec<EngineBatchItem> = job
            .conditioning_inputs
            .iter()
            .map(|input| engine_batch_item_for_input(&job, input))
            .collect();

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].stem, "sample_0_L");
        assert_eq!(items[1].stem, "sample_0_R");
        assert_eq!(items[0].source_channels, 2);
        assert_eq!(items[0].conditioning_channels, 1);
        assert_eq!(items[0].channel_index, 0);
        assert_eq!(items[0].channel_name, "left");
        assert_eq!(items[1].channel_index, 1);
        assert_eq!(items[1].channel_name, "right");
    }

    #[test]
    fn clear_cache_removes_all_cleanup_variants() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("Should create temp home");
        let _home = HomeGuard::set(temp_home.path());
        let cache = cache_dir();
        std::fs::create_dir_all(&cache).expect("cache dir should exist");

        let data = std::fs::read(BASIC_FIXTURE).expect("fixture should load");
        let mut module = Module::from_memory(&data).expect("fixture module should load");

        // Use a single eligible sample — enough to verify all cleanup variant
        // hashes are produced and deleted. Processing all samples is unnecessary
        // and dominated the old test's runtime.
        let sample = read_raw_samples(&mut module)
            .into_iter()
            .find(|s| !s.data.is_empty())
            .expect("fixture should have at least one eligible sample");
        let loop_prep = LoopPrepPlan::from_sample(sample.source_length_frames, sample.loop_info);

        let mut hashes =
            collect_cache_hashes_for_sample(&sample.data, sample.rate, sample.channels, loop_prep);
        hashes.sort();
        hashes.dedup();
        assert!(!hashes.is_empty(), "fixture should produce cache hashes");

        // Create dummy cache files for every hash
        for hash in &hashes {
            let path = cache.join(format!("{hash}-audiosr.flac"));
            std::fs::write(&path, b"cache").expect("test cache entry should write");
        }

        // Delete using the same hashes
        let deleted = delete_cache_files_for_hashes(&hashes);
        assert!(deleted >= hashes.len());
        for hash in &hashes {
            assert!(
                !cache.join(format!("{hash}-audiosr.flac")).exists(),
                "cache file for hash {hash} should be removed",
            );
        }
    }

    // Build a mono sine-like ramp with N frames, amplitude `amp`. Each frame
    // differs from the previous by roughly `amp / period`, producing a steady
    // slope whose |value| is `amp / period`.
    fn ramp_signal(frames: usize, amp: f64, period: f64) -> Vec<f64> {
        (0..frames)
            .map(|i| amp * (i as f64 / period).sin())
            .collect()
    }

    #[test]
    fn loop_body_slope_envelope_catches_transient_inside_body() {
        let frames = 256usize;
        let mut data = ramp_signal(frames, 0.3, 32.0);
        // Insert a sharp spike at frame 128.
        data[128] = 0.9;
        data[129] = -0.9;
        let env = loop_body_slope_envelope(&data, 1, 0, frames, 8);
        assert_eq!(env.len(), 8);
        // The window containing frame 128 should have a much higher peak slope
        // than the others.
        let max_idx = env
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Windows are 32 pairs each (255 pairs / 8 ≈ 31); spike at pair 128
        // lives in window 4.
        assert_eq!(max_idx, 4, "spike window should have max slope");
        assert!(env[max_idx] > 1.0, "spike slope should be large");
        // Neighboring windows' peak slope should be much smaller.
        for (i, v) in env.iter().enumerate() {
            if i != max_idx {
                assert!(*v < 0.1, "window {i} should not see the spike: {v}");
            }
        }
    }

    #[test]
    fn loop_body_slope_envelope_catches_end_burst() {
        let frames = 184usize;
        let mut ai = ramp_signal(frames, 0.2, 46.0);
        // Burst in the last window (index 5 of 6).
        for f in 160..frames {
            ai[f] = if f % 2 == 0 { 0.9 } else { -0.9 };
        }
        let orig = ramp_signal(frames, 0.2, 46.0);
        let ai_env = loop_body_slope_envelope(&ai, 1, 0, frames, 6);
        let orig_env = loop_body_slope_envelope(&orig, 1, 0, frames, 6);
        // First 5 windows should be comparable; last window should differ
        // massively.
        for i in 0..5 {
            let ratio = ai_env[i] / orig_env[i].max(1e-9);
            assert!(
                ratio < 2.0,
                "window {i} should not fire: ai={} orig={}",
                ai_env[i],
                orig_env[i]
            );
        }
        let ratio_last = ai_env[5] / orig_env[5].max(1e-9);
        assert!(
            ratio_last > 10.0,
            "last window should show massive slope spike: ai={} orig={} ratio={}",
            ai_env[5],
            orig_env[5],
            ratio_last,
        );
    }

    #[test]
    fn loop_body_slope_envelope_matches_identical_signals() {
        let frames = 200usize;
        let data = ramp_signal(frames, 0.5, 32.0);
        let env_a = loop_body_slope_envelope(&data, 1, 0, frames, 8);
        let env_b = loop_body_slope_envelope(&data, 1, 0, frames, 8);
        assert_eq!(env_a, env_b);
    }

    #[test]
    fn loop_body_slope_envelope_respects_floor() {
        // Near-silent signals should yield sub-floor peaks.
        let frames = 128usize;
        let quiet: Vec<f64> = (0..frames).map(|_| 0.0).collect();
        let env = loop_body_slope_envelope(&quiet, 1, 0, frames, 4);
        assert_eq!(env.len(), 4);
        for v in env {
            assert!(v < AI_LOOP_BODY_SLOPE_FLOOR);
        }
    }

    #[test]
    fn loop_body_slope_envelope_handles_short_loops() {
        // Tiny loop — 1 window minimum.
        let data: Vec<f64> = vec![0.0, 0.1, 0.05, 0.2, 0.1];
        let env = loop_body_slope_envelope(&data, 1, 0, 5, 1);
        assert_eq!(env.len(), 1);
        assert!(env[0] > 0.0);

        // Degenerate cases.
        assert!(loop_body_slope_envelope(&[], 1, 0, 0, 4).is_empty());
        assert!(loop_body_slope_envelope(&data, 0, 0, 5, 4).is_empty());
        assert!(loop_body_slope_envelope(&data, 1, 3, 3, 4).is_empty());
    }

    // ---------- Mixed sustain + normal loop tiling ----------

    fn build_mixed_sample_data() -> (Vec<f64>, SampleLoopInfo) {
        // Layout: [attack=16][sustain_body=20][middle=8][normal_body=24][tail=16] = 84
        let mut data: Vec<f64> = Vec::new();
        data.resize(16, 0.1);
        data.resize(36, 0.2);
        data.resize(44, 0.3);
        data.resize(68, 0.4);
        data.resize(84, 0.5);
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(44, 68),
            SampleLoopRegion::forward(16, 36),
        );
        (data, loop_info)
    }

    #[test]
    fn pad_for_engine_tiles_mixed_forward_sustain_forward_normal() {
        let (data, loop_info) = build_mixed_sample_data();
        let (padded, layout) = pad_for_engine(&data, 1, 96, loop_info);
        let mixed = layout
            .expect("mixed sustain+normal loops should produce a layout")
            .expect_mixed();

        assert!(mixed.sustain_block.body_copies >= 3);
        assert!(mixed.normal_block.body_copies >= 3);
        assert_eq!(mixed.base_timeline_frames, 84);
        assert_eq!(mixed.sustain_block.offset_frames, 84);
        // Sustain block = attack(16) + copies * sustain_body(20) + tail_after_sustain(48)
        let sustain_block_len = 16 + mixed.sustain_block.body_copies * 20 + 48;
        assert_eq!(
            mixed.normal_block.offset_frames,
            (84 + sustain_block_len) as i64,
        );
        let normal_block_len = 44 + mixed.normal_block.body_copies * 24 + 16;
        assert_eq!(padded.len(), 84 + sustain_block_len + normal_block_len);
        assert_eq!(
            &padded[..84],
            data.as_slice(),
            "base timeline must be verbatim"
        );
        assert!(
            padded.len() >= 96,
            "combined padded buffer must meet min_samples"
        );
    }

    #[test]
    fn pad_for_engine_tiles_mixed_forward_sustain_pingpong_normal() {
        let (data, _) = build_mixed_sample_data();
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::ping_pong(44, 68),
            SampleLoopRegion::forward(16, 36),
        );
        let (_padded, layout) = pad_for_engine(&data, 1, 32, loop_info);
        let mixed = layout.expect("mixed").expect_mixed();
        assert!(mixed.sustain_block.body_copies >= 3);
        assert!(mixed.normal_block.body_copies >= 5);
        assert_eq!(mixed.normal_block.body_copies % 4, 1);
    }

    #[test]
    fn pad_for_engine_tiles_mixed_pingpong_sustain_forward_normal() {
        let (data, _) = build_mixed_sample_data();
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(44, 68),
            SampleLoopRegion::ping_pong(16, 36),
        );
        let (_padded, layout) = pad_for_engine(&data, 1, 32, loop_info);
        let mixed = layout.expect("mixed").expect_mixed();
        assert!(mixed.sustain_block.body_copies >= 5);
        assert_eq!(mixed.sustain_block.body_copies % 4, 1);
        assert!(mixed.normal_block.body_copies >= 3);
    }

    #[test]
    fn pad_for_engine_tiles_mixed_pingpong_sustain_pingpong_normal() {
        let (data, _) = build_mixed_sample_data();
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::ping_pong(44, 68),
            SampleLoopRegion::ping_pong(16, 36),
        );
        let (_padded, layout) = pad_for_engine(&data, 1, 32, loop_info);
        let mixed = layout.expect("mixed").expect_mixed();
        assert!(mixed.sustain_block.body_copies >= 5);
        assert_eq!(mixed.sustain_block.body_copies % 4, 1);
        assert!(mixed.normal_block.body_copies >= 5);
        assert_eq!(mixed.normal_block.body_copies % 4, 1);
    }

    #[test]
    fn pad_for_engine_full_containment_falls_back_to_single_sustain() {
        // normal [10, 20] is fully inside sustain [5, 25]; mixed path is
        // avoided because the normal block would be entirely overwritten by
        // sustain in the extractor.
        let data: Vec<f64> = (0..32).map(|i| i as f64 * 0.01).collect();
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(10, 20),
            SampleLoopRegion::forward(5, 25),
        );
        let (_padded, layout) = pad_for_engine(&data, 1, 64, loop_info);
        let single = layout
            .expect("full-containment should still tile (via sustain)")
            .expect_single();
        assert_eq!(single.loop_source, TiledLoopSource::Sustain);
    }

    #[test]
    fn normal_exclusive_ranges_handles_the_four_plan_cases() {
        // Non-overlap: sustain entirely before normal.
        assert_eq!(normal_exclusive_ranges(20, 30, 5, 15), vec![(20, 30)],);
        // Partial overlap: normal.start < sustain.end < normal.end with
        // sustain starting before normal.
        assert_eq!(normal_exclusive_ranges(10, 20, 5, 15), vec![(15, 20)],);
        // Full containment: normal ⊂ sustain → empty (defensive; mixed path
        // falls back to single sustain).
        assert_eq!(
            normal_exclusive_ranges(12, 18, 10, 20),
            Vec::<(usize, usize)>::new(),
        );
        // Normal contains sustain: two flanking pieces.
        assert_eq!(
            normal_exclusive_ranges(5, 25, 10, 20),
            vec![(5, 10), (20, 25)],
        );
    }

    #[test]
    fn extract_channel_result_mixed_preserves_length_and_reconstructs_bodies() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("mixed.wav");
        let rate = 48_000u32;
        let (data, loop_info) = build_mixed_sample_data();

        let (padded, layout) = pad_for_engine(&data, 1, 96, loop_info);
        let mixed = layout.expect("mixed").expect_mixed();

        // Identity engine: the padded conditioning buffer IS the engine output.
        write_wav(&output_wav, &padded, rate, 1).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "MixedFF".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: 84,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("in.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: 84,
            engine_input_layout: Some(EngineInputLayout::Mixed(mixed)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result =
            extract_result(&job, &output_wav, "Identity").expect("mixed extract should succeed");
        assert_eq!(result.length_frames, 84);
        assert_eq!(result.data.len(), 84);

        // Region centers, chosen well away from the 16-frame seam windows.
        assert!((result.data[7] - 0.1).abs() < 1e-9, "attack center");
        assert!((result.data[26] - 0.2).abs() < 1e-9, "sustain body center");
        assert!((result.data[56] - 0.4).abs() < 1e-9, "normal body center");
        assert!((result.data[76] - 0.5).abs() < 1e-9, "tail center");
    }

    #[test]
    fn extract_channel_result_mixed_sustain_owns_overlap() {
        // Overlap case: sustain [16, 48], normal [32, 80]; overlap = [32, 48].
        // Each tiled-block middle-body gets a distinctive marker and the test
        // picks assertion frames outside all 16-frame seam fade windows.
        // Seam zones (half=8): 16 → 8..24, 48 → 40..56, 80 → 72..88.
        // Clean interiors: sustain-exclusive [24, 32], overlap [32, 40],
        // normal-exclusive [56, 72].
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("overlap.wav");
        let rate = 48_000u32;

        let mut data: Vec<f64> = Vec::new();
        data.resize(16, 0.05);
        data.resize(48, 0.25);
        data.resize(80, 0.45);
        data.resize(96, 0.65);
        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(32, 80),
            SampleLoopRegion::forward(16, 48),
        );
        let (mut padded, layout) = pad_for_engine(&data, 1, 96, loop_info);
        let mixed = layout.expect("mixed").expect_mixed();

        // Overwrite each middle body with a marker so we can tell which loop
        // the extracted overlap comes from.
        let s_body_frames = 48 - 16; // 32
        let n_body_frames = 80 - 32; // 48
        let s_middle_start = mixed.sustain_block.offset_frames as usize
            + 16
            + (mixed.sustain_block.body_copies / 2) * s_body_frames;
        let n_middle_start = mixed.normal_block.offset_frames as usize
            + 32
            + (mixed.normal_block.body_copies / 2) * n_body_frames;
        for f in 0..s_body_frames {
            padded[s_middle_start + f] = 1.0;
        }
        for f in 0..n_body_frames {
            padded[n_middle_start + f] = -1.0;
        }

        write_wav(&output_wav, &padded, rate, 1).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "Overlap".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: 96,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("in.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: 96,
            engine_input_layout: Some(EngineInputLayout::Mixed(mixed)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result =
            extract_result(&job, &output_wav, "Identity").expect("overlap extract should succeed");
        assert_eq!(result.length_frames, 96);

        // Sustain-exclusive clean interior: [24, 32).
        assert!(
            (result.data[28] - 1.0).abs() < 1e-9,
            "sustain-exclusive must be sustain marker, got {}",
            result.data[28],
        );
        // Overlap clean interior: [32, 40). Sustain owns overlap.
        assert!(
            (result.data[36] - 1.0).abs() < 1e-9,
            "overlap must be owned by sustain, got {}",
            result.data[36],
        );
        // Normal-exclusive clean interior: [56, 72).
        assert!(
            (result.data[64] + 1.0).abs() < 1e-9,
            "normal-exclusive must be normal marker, got {}",
            result.data[64],
        );
    }

    #[test]
    fn extract_sample_jobs_writes_single_conditioning_wav_for_mixed_loops() {
        let head = 48usize;
        let sustain_body = 48usize;
        let mid = 24usize;
        let normal_body = 64usize;
        let tail = 24usize;
        let total = head + sustain_body + mid + normal_body + tail;

        let mut original: Vec<f64> = Vec::new();
        original.resize(head, 0.1);
        original.resize(head + sustain_body, 0.2);
        original.resize(head + sustain_body + mid, 0.3);
        original.resize(head + sustain_body + mid + normal_body, 0.4);
        original.resize(total, 0.5);

        let sample = OriginalSample {
            index: 0,
            data: original.clone(),
            rate: 24_000,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: total as i64,
            effective_length_frames: total as i64,
            loop_start_frames: head as i64,
            looped: true,
            loop_info: SampleLoopInfo::with_loops(
                SampleLoopRegion::forward(
                    (head + sustain_body + mid) as i64,
                    (head + sustain_body + mid + normal_body) as i64,
                ),
                SampleLoopRegion::forward(head as i64, (head + sustain_body) as i64),
            ),
            name: "Mixed".into(),
        };
        let cancel_flag = AtomicBool::new(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let jobs = extract_sample_jobs(
            &[sample],
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("jobs");
        let job = jobs.first().expect("one job");

        // One conditioning WAV for mono mixed input.
        assert_eq!(job.conditioning_inputs.len(), 1);

        let mixed = job
            .engine_input_layout
            .expect("mixed layout must be recorded on job")
            .expect_mixed();
        assert!(mixed.sustain_block.body_copies >= 3);
        assert!(mixed.normal_block.body_copies >= 3);

        // input_length_frames should match the actual WAV frame count.
        let (buf, _channels) =
            read_wav(&job.conditioning_inputs[0].input_path).expect("decode conditioning wav");
        assert_eq!(
            sample_frame_count(&buf, 1),
            job.conditioning_inputs[0].input_length_frames,
        );
        // And that length equals base_timeline + both tiled blocks at conditioning rate.
        assert!(
            job.conditioning_inputs[0].input_length_frames > mixed.base_timeline_frames,
            "mixed conditioning input must be larger than the base timeline alone",
        );
    }

    #[test]
    fn extract_channel_result_mixed_handles_normal_before_sustain_overlap() {
        // "Post-keyoff wrap" case from the plan: normal loop starts before
        // sustain and overlaps it. Layout: sustain [40, 100], normal [8, 56],
        // overlap = [40, 56]. Normal-exclusive = [8, 40]. The sustain entry
        // seam at 40 is shared with the normal exit seam at 40 — the
        // extractor skips the sustain entry seam so the transition is owned
        // by the surviving normal exit seam.
        //
        // Seam zones (half = 8):
        //   normal seams:  8 → [0, 16],   40 → [32, 48]
        //   sustain seams: 40 skipped,    100 → [92, 108]
        // Clean interiors:
        //   normal-exclusive: [16, 32]  → assert at frame 24
        //   overlap:          [48, 56]  → assert at frame 52
        //   sustain-exclusive:[56, 92]  → assert at frame 70
        let work_dir = tempfile::tempdir().expect("tempdir");
        let output_wav = work_dir.path().join("wrap.wav");
        let rate = 48_000u32;

        let mut data: Vec<f64> = Vec::new();
        data.resize(8, 0.05);
        data.resize(56, 0.25);
        data.resize(100, 0.45);
        data.resize(120, 0.65);

        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(8, 56),
            SampleLoopRegion::forward(40, 100),
        );
        let (mut padded, layout) = pad_for_engine(&data, 1, 120, loop_info);
        let mixed = layout.expect("mixed").expect_mixed();

        let s_body = 100 - 40;
        let n_body = 56 - 8;
        let s_middle_start = mixed.sustain_block.offset_frames as usize
            + 40
            + (mixed.sustain_block.body_copies / 2) * s_body;
        let n_middle_start = mixed.normal_block.offset_frames as usize
            + 8
            + (mixed.normal_block.body_copies / 2) * n_body;
        for f in 0..s_body {
            padded[s_middle_start + f] = 1.0;
        }
        for f in 0..n_body {
            padded[n_middle_start + f] = -1.0;
        }

        write_wav(&output_wav, &padded, rate, 1).expect("write wav");

        let job = SampleJob {
            index: 0,
            name: "Wrap".into(),
            original_data: data.clone(),
            rate: rate as i32,
            output_sample_rate_hz: rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: 120,
            looped: true,
            loop_info,
            conditioning_inputs: mono_conditioning_inputs(
                work_dir.path().join("in.wav"),
                padded.len() as i64,
            ),
            conditioning_rate_hz: rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: data.clone(),
            original_length_48k_frames: 120,
            engine_input_layout: Some(EngineInputLayout::Mixed(mixed)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let result =
            extract_result(&job, &output_wav, "Identity").expect("wrap extract should succeed");
        assert_eq!(result.length_frames, 120);
        assert!(
            (result.data[24] + 1.0).abs() < 1e-9,
            "normal-exclusive clean interior must be normal marker, got {}",
            result.data[24],
        );
        assert!(
            (result.data[52] - 1.0).abs() < 1e-9,
            "overlap must be owned by sustain, got {}",
            result.data[52],
        );
        assert!(
            (result.data[70] - 1.0).abs() < 1e-9,
            "sustain-exclusive clean interior must be sustain marker, got {}",
            result.data[70],
        );
    }

    #[test]
    fn extract_sample_jobs_writes_stereo_conditioning_wavs_for_mixed_loops() {
        // Stereo mixed-loop sample: each channel gets its own conditioning
        // WAV, both carry EngineInputLayout::Mixed, and input_length_frames
        // matches each channel's WAV.
        let channels = 2;
        let head = 48usize;
        let sustain_body = 48usize;
        let mid = 24usize;
        let normal_body = 64usize;
        let tail = 24usize;
        let total = head + sustain_body + mid + normal_body + tail;

        let mut original: Vec<f64> = Vec::with_capacity(total * channels);
        for f in 0..total {
            let l = match f {
                x if x < head => 0.1,
                x if x < head + sustain_body => 0.2,
                x if x < head + sustain_body + mid => 0.3,
                x if x < head + sustain_body + mid + normal_body => 0.4,
                _ => 0.5,
            };
            original.push(l);
            original.push(-l);
        }

        let sample = OriginalSample {
            index: 0,
            data: original,
            rate: 24_000,
            channels: channels as i32,
            bits_per_sample: 16,
            source_length_frames: total as i64,
            effective_length_frames: total as i64,
            loop_start_frames: head as i64,
            looped: true,
            loop_info: SampleLoopInfo::with_loops(
                SampleLoopRegion::forward(
                    (head + sustain_body + mid) as i64,
                    (head + sustain_body + mid + normal_body) as i64,
                ),
                SampleLoopRegion::forward(head as i64, (head + sustain_body) as i64),
            ),
            name: "StereoMixed".into(),
        };
        let cancel_flag = AtomicBool::new(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let jobs = extract_sample_jobs(
            &[sample],
            work_dir.path(),
            0.01,
            off_settings(),
            &cancel_flag,
        )
        .expect("jobs");
        let job = jobs.first().expect("one job");

        assert_eq!(job.conditioning_inputs.len(), 2);
        let mixed = job
            .engine_input_layout
            .expect("mixed layout must be recorded on stereo job")
            .expect_mixed();
        assert!(mixed.sustain_block.body_copies >= 3);
        assert!(mixed.normal_block.body_copies >= 3);

        for channel_input in &job.conditioning_inputs {
            let (buf, wav_channels) =
                read_wav(&channel_input.input_path).expect("decode channel wav");
            assert_eq!(wav_channels, 1, "stereo inputs split into mono WAVs");
            assert_eq!(
                sample_frame_count(&buf, 1),
                channel_input.input_length_frames,
            );
        }
    }

    #[test]
    fn compute_engine_cache_key_pipeline_version_invalidates_caches() {
        // Same source + engine args produce different keys under different
        // pipeline versions, so a tiling/extraction change that bumps
        // ENGINE_PIPELINE_VERSION forces cache misses on pre-bump entries.
        let pcm = [0x42u8; 32];
        let make = |ver: u8| -> String {
            compute_engine_cache_key(&pcm, 22_050, 16, 100, 200, 1, 48_000, "audiosr-v1", 8, ver)
        };

        let v0 = make(0);
        let v1 = make(1);
        let v1_again = make(1);
        let v2 = make(2);

        assert_ne!(v0, v1, "bumping pipeline_version must change the key");
        assert_ne!(v1, v2, "each bump must change the key");
        assert_eq!(v1, v1_again, "same inputs produce stable keys");
    }

    #[test]
    fn extract_mixed_channel_output_does_not_zero_pad_at_odd_native_rates() {
        // Two-step scaling diverges from one-step for certain source sizes at
        // odd native rates. With native = 11025 Hz and a 500-frame source:
        //   cond (24k): round(500 · 24000 / 11025) = 1088
        //   base_timeline_out (24k → 48k): 1088 · 2 = 2176
        //   trim_frames (11025 → 48k direct): 2177
        // The mixed extractor must return a 2176-frame output rather than
        // silently zero-padding one frame to reach `trim_frames`, matching
        // the single-loop extractor's behavior on short engine output.
        let native_rate = 11025u32;
        let cond_rate = 24_000u32;
        let output_rate = 48_000u32;
        let source_frames_native = 500usize;
        let base_timeline_cond = scaled_frame_count(source_frames_native, native_rate, cond_rate);
        let trim_frames = scaled_frame_count(source_frames_native, native_rate, output_rate);
        let base_timeline_out = scaled_frame_count(base_timeline_cond, cond_rate, output_rate);
        assert!(
            base_timeline_out < trim_frames,
            "test setup precondition: base_timeline_out ({base_timeline_out}) should be < \
             trim_frames ({trim_frames}) at native={native_rate} / cond={cond_rate} / \
             output={output_rate}",
        );

        let loop_info = SampleLoopInfo::with_loops(
            SampleLoopRegion::forward(300, 400),
            SampleLoopRegion::forward(100, 200),
        );
        // Synthetic mixed layout: both blocks start after the base timeline
        // and are just long enough that the tiled block offsets stay inside
        // raw_data. Exact block sizes don't matter for this test — we only
        // check the output length and that no zero padding is emitted.
        let sustain_block = TiledBlock {
            offset_frames: base_timeline_cond as i64,
            body_copies: 3,
        };
        let normal_block = TiledBlock {
            offset_frames: (base_timeline_cond + 400) as i64,
            body_copies: 3,
        };
        let mixed = MixedTilingLayout {
            base_timeline_frames: base_timeline_cond as i64,
            sustain_block,
            normal_block,
        };

        // Fill raw_data (at output rate) with a sentinel so we can spot any
        // zero-padded trailing frame. Size it generously so block reads stay
        // in bounds.
        const SENTINEL: f64 = 0.42;
        let raw_frames = base_timeline_out + scaled_frame_count(1200, cond_rate, output_rate);
        let raw_data: Vec<f64> = vec![SENTINEL; raw_frames];

        let job = SampleJob {
            index: 0,
            name: "OddRateMixed".into(),
            original_data: Vec::new(),
            rate: native_rate as i32,
            output_sample_rate_hz: output_rate as i32,
            channels: 1,
            bits_per_sample: 16,
            source_length_frames: source_frames_native as i64,
            looped: true,
            loop_info,
            conditioning_inputs: Vec::new(),
            conditioning_rate_hz: cond_rate,
            pcm_sha256: [0u8; 32],
            target_rms: 0.0,
            reference_48k: Vec::new(),
            original_length_48k_frames: trim_frames as i64,
            engine_input_layout: Some(EngineInputLayout::Mixed(mixed)),
            native_inputs: Vec::new(),
            native_rate_hz: 0,
            native_input_layout: None,
        };

        let extracted = extract_mixed_channel_output(
            &raw_data,
            1,
            &job,
            mixed,
            trim_frames,
            job.conditioning_rate_hz,
            0,
        );

        assert_eq!(
            extracted.len(),
            base_timeline_out,
            "mixed extractor should emit {base_timeline_out} frames (matching the \
             base timeline at output rate), not zero-pad to trim_frames ({trim_frames})",
        );
        // Seam crossfades introduce tiny float-arithmetic noise even when both
        // sides are the sentinel; assert a narrow window around SENTINEL to
        // confirm no frame collapsed to zero.
        for (i, v) in extracted.iter().enumerate() {
            assert!(
                (v - SENTINEL).abs() < 1e-9,
                "frame {i} = {v} diverged from sentinel {SENTINEL} — possible zero-pad hole",
            );
        }
    }

    #[test]
    fn detect_with_fallback_partitions_engines_correctly() {
        let engine = RemasterEngine::from_test_engines_with_fallback(
            vec![
                Box::new(DummyEngine {
                    name: "AudioSR",
                    cache_id: "audiosr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "LavaSR",
                    cache_id: "lavasr-v0.1",
                }),
            ],
            vec![Box::new(DummyEngine {
                name: "FLowHigh",
                cache_id: "flowhigh-v0.1",
            })],
        );

        assert_eq!(
            engine.engines.iter().map(|e| e.name()).collect::<Vec<_>>(),
            vec!["AudioSR", "LavaSR"],
        );
        assert_eq!(
            engine
                .fallback_engines
                .iter()
                .map(|e| e.name())
                .collect::<Vec<_>>(),
            vec!["FLowHigh"],
        );
        assert_eq!(engine.available_engine_names(), vec!["AudioSR", "LavaSR"]);
    }

    #[test]
    fn sample_has_gate_failure_false_for_self_correlated_candidate() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let job = sample_job(work_dir.path());
        let data = job.reference_48k.clone();
        let frames = sample_frame_count(&data, 1);
        let candidates = vec![SampleResult {
            index: job.index,
            data,
            length_frames: frames,
            channels: 1,
            sample_rate_hz: 48_000,
            engine_name: "AudioSR".to_string(),
            discovered_loops: None,
        }];
        assert!(!sample_has_gate_failure(&job, &candidates));
    }

    #[test]
    fn sample_has_gate_failure_true_for_uncorrelated_candidate() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let job = sample_job(work_dir.path());
        // Candidate: unrelated spectrum (harmonic stack at non-overlapping
        // frequencies + noise) — correlates poorly with the 220 Hz source.
        let data: Vec<f64> = (0..48_000)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (t * 2.0 * std::f64::consts::PI * 5000.0).sin() * 0.4
                    + (t * 2.0 * std::f64::consts::PI * 7000.0).cos() * 0.3
                    + (i as f64 * 0.017).sin() * 0.2
            })
            .collect();
        let frames = sample_frame_count(&data, 1);
        let candidates = vec![SampleResult {
            index: job.index,
            data,
            length_frames: frames,
            channels: 1,
            sample_rate_hz: 48_000,
            engine_name: "AudioSR".to_string(),
            discovered_loops: None,
        }];
        assert!(sample_has_gate_failure(&job, &candidates));
    }

    #[test]
    fn snapshot_failing_samples_bumps_engines_total_and_returns_failing_only() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        // Two jobs; sample 0 has a low-score candidate, sample 1 all high.
        let mut job_ok = sample_job(work_dir.path());
        job_ok.index = 1;
        let job_bad = sample_job(work_dir.path());

        let good_candidate = |job: &SampleJob, name: &str| SampleResult {
            index: job.index,
            data: job.reference_48k.clone(),
            length_frames: sample_frame_count(&job.reference_48k, 1),
            channels: 1,
            sample_rate_hz: 48_000,
            engine_name: name.to_string(),
            discovered_loops: None,
        };
        let bad_candidate = |job: &SampleJob, name: &str| {
            let data: Vec<f64> = (0..48_000)
                .map(|i| {
                    let t = i as f64 / 48_000.0;
                    (t * 2.0 * std::f64::consts::PI * 5000.0).sin() * 0.4
                        + (i as f64 * 0.017).sin() * 0.3
                })
                .collect();
            let frames = sample_frame_count(&data, 1);
            SampleResult {
                index: job.index,
                data,
                length_frames: frames,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: name.to_string(),
                discovered_loops: None,
            }
        };

        let pending: PendingOutputMap =
            std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        {
            let mut map = pending.lock().unwrap();
            map.insert(
                job_bad.index,
                PendingSampleOutputs {
                    candidates: vec![
                        bad_candidate(&job_bad, "AudioSR"),
                        good_candidate(&job_bad, "LavaSR"),
                    ],
                    engines_done: 2,
                    engines_total: 2,
                    last_emitted_candidate_count: 0,
                },
            );
            map.insert(
                job_ok.index,
                PendingSampleOutputs {
                    candidates: vec![
                        good_candidate(&job_ok, "AudioSR"),
                        good_candidate(&job_ok, "LavaSR"),
                    ],
                    engines_done: 2,
                    engines_total: 2,
                    last_emitted_candidate_count: 0,
                },
            );
        }

        let jobs = vec![job_bad.clone(), job_ok.clone()];
        let fallback: Vec<Box<dyn UpsampleEngine>> = vec![Box::new(DummyEngine {
            name: "FLowHigh",
            cache_id: "flowhigh-v0.1",
        })];

        let info = snapshot_failing_samples_and_bump_totals(&pending, &jobs, &fallback);

        assert_eq!(info.failing_sample_indices, vec![job_bad.index]);
        // job_bad is at index 0 in `jobs`, job_ok at index 1.
        assert_eq!(info.counts_by_job, vec![1, 0]);
        assert_eq!(info.total_additional_tasks, 1);
        let map = pending.lock().unwrap();
        assert_eq!(map[&job_bad.index].engines_total, 3);
        assert_eq!(map[&job_ok.index].engines_total, 2);
    }

    #[test]
    fn fallback_wave_skipped_when_all_primary_pass_gate() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("temp work");
        let job = sample_job(work_dir.path());

        // Cache both primaries with self-correlated (high-score) candidates.
        // FLowHigh in fallback is NOT cached; if the fallback wave runs it
        // would try to spawn (DummyEngine panics), failing the test.
        for (cache_id, name) in [("audiosr-v0.1", "AudioSR"), ("lavasr-v0.1", "LavaSR")] {
            cache_store(
                &job.pcm_sha256,
                cache_id,
                50,
                &SampleResult {
                    index: 0,
                    data: job.reference_48k.clone(),
                    length_frames: sample_frame_count(&job.reference_48k, 1),
                    channels: 1,
                    sample_rate_hz: 48_000,
                    engine_name: name.to_string(),
                    discovered_loops: None,
                },
                1.0,
            );
        }

        let engine = RemasterEngine::from_test_engines_with_fallback(
            vec![
                Box::new(DummyEngine {
                    name: "AudioSR",
                    cache_id: "audiosr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "LavaSR",
                    cache_id: "lavasr-v0.1",
                }),
            ],
            vec![Box::new(DummyEngine {
                name: "FLowHigh",
                cache_id: "flowhigh-v0.1",
            })],
        );
        let (progress_tx, _progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("temp remaster"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                false,
                false,
            )
            .expect("remaster");

        let candidates: Vec<String> = result_rx
            .try_iter()
            .filter_map(|o| match o {
                RemasterOutput::Candidate(c) => Some(c.engine_name),
                RemasterOutput::Final(_) => None,
            })
            .collect();
        assert_eq!(
            candidates.len(),
            2,
            "FLowHigh should NOT run — all primaries passed the gate. Got: {candidates:?}",
        );
        assert!(candidates.contains(&"AudioSR".to_string()));
        assert!(candidates.contains(&"LavaSR".to_string()));
        assert!(
            !candidates.contains(&"FLowHigh".to_string()),
            "fallback engine ran when no primary failed the gate",
        );
    }

    #[test]
    fn fallback_wave_runs_when_primary_candidate_fails_gate() {
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("temp work");
        let job = sample_job(work_dir.path());

        // AudioSR: LOW-score (uncorrelated) — will trip the gate.
        // LavaSR: HIGH-score (self-correlated).
        // FLowHigh (fallback): HIGH-score — should run because AudioSR failed.
        let low_score_data: Vec<f64> = (0..48_000)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (t * 2.0 * std::f64::consts::PI * 5000.0).sin() * 0.4
                    + (i as f64 * 0.017).sin() * 0.3
            })
            .collect();

        cache_store(
            &job.pcm_sha256,
            "audiosr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: low_score_data,
                length_frames: 48_000,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );
        for (cache_id, name) in [("lavasr-v0.1", "LavaSR"), ("flowhigh-v0.1", "FLowHigh")] {
            cache_store(
                &job.pcm_sha256,
                cache_id,
                50,
                &SampleResult {
                    index: 0,
                    data: job.reference_48k.clone(),
                    length_frames: sample_frame_count(&job.reference_48k, 1),
                    channels: 1,
                    sample_rate_hz: 48_000,
                    engine_name: name.to_string(),
                    discovered_loops: None,
                },
                1.0,
            );
        }

        let engine = RemasterEngine::from_test_engines_with_fallback(
            vec![
                Box::new(DummyEngine {
                    name: "AudioSR",
                    cache_id: "audiosr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "LavaSR",
                    cache_id: "lavasr-v0.1",
                }),
            ],
            vec![Box::new(DummyEngine {
                name: "FLowHigh",
                cache_id: "flowhigh-v0.1",
            })],
        );
        let (progress_tx, _progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("temp remaster"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                false,
                false,
            )
            .expect("remaster");

        let outputs: Vec<RemasterOutput> = result_rx.try_iter().collect();
        let candidates: Vec<String> = outputs
            .iter()
            .filter_map(|o| match o {
                RemasterOutput::Candidate(c) => Some(c.engine_name.clone()),
                RemasterOutput::Final(_) => None,
            })
            .collect();
        assert_eq!(
            candidates.len(),
            3,
            "FLowHigh fallback should run because AudioSR failed the gate. Got: {candidates:?}",
        );
        assert!(candidates.contains(&"FLowHigh".to_string()));

        // Strongest assertion: the last Final's consensus must include the
        // fallback engine as a contributor. The short code for FLowHigh is
        // "F" (see `engine_short_code`).
        let last_final_name = outputs
            .iter()
            .rev()
            .find_map(|o| match o {
                RemasterOutput::Final(result) => Some(result.engine_name.clone()),
                RemasterOutput::Candidate(_) => None,
            })
            .expect("expected at least one Final");
        assert!(
            last_final_name.starts_with("Quinlight Audio") && last_final_name.contains('F'),
            "last Final should be a Quinlight consensus that includes FLowHigh; got {last_final_name:?}",
        );
    }

    #[test]
    fn fallback_wave_respects_supports_original_rate() {
        // A fallback engine that doesn't support the job's original rate must
        // be silently excluded: no candidate emitted, engines_total not bumped,
        // and its panicking spawn_batch never called.
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("temp work");
        let job = sample_job(work_dir.path()); // rate = 16_000

        // Primary AudioSR: low-score (triggers fallback).
        // Primary LavaSR: high-score.
        let low_score_data: Vec<f64> = (0..48_000)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (t * 2.0 * std::f64::consts::PI * 5000.0).sin() * 0.4
                    + (i as f64 * 0.017).sin() * 0.3
            })
            .collect();
        cache_store(
            &job.pcm_sha256,
            "audiosr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: low_score_data,
                length_frames: 48_000,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );
        cache_store(
            &job.pcm_sha256,
            "lavasr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: job.reference_48k.clone(),
                length_frames: sample_frame_count(&job.reference_48k, 1),
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "LavaSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );

        // Fallback: a rate-limited engine capped below the job's 16 kHz
        // native rate. If `supports_original_rate` is respected the wave is
        // a no-op; if not, `spawn_batch`'s `unreachable!()` will panic.
        let engine = RemasterEngine::from_test_engines_with_fallback(
            vec![
                Box::new(DummyEngine {
                    name: "AudioSR",
                    cache_id: "audiosr-v0.1",
                }),
                Box::new(DummyEngine {
                    name: "LavaSR",
                    cache_id: "lavasr-v0.1",
                }),
            ],
            vec![Box::new(RateLimitedEngine {
                name: "FLowHigh",
                max_original_rate_hz: 8_000,
            })],
        );

        let (progress_tx, progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(false);

        engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("temp remaster"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                false,
                false,
            )
            .expect("remaster");

        let candidate_names: Vec<String> = result_rx
            .try_iter()
            .filter_map(|o| match o {
                RemasterOutput::Candidate(c) => Some(c.engine_name),
                RemasterOutput::Final(_) => None,
            })
            .collect();
        assert_eq!(candidate_names.len(), 2);
        assert!(
            !candidate_names.iter().any(|n| n == "FLowHigh"),
            "rate-incompatible fallback engine must not produce candidates",
        );

        // `EngineProgress` engines_total should stay at 2 throughout — no
        // rate-eligible fallback engine means no bump.
        let max_engines_total = progress_rx
            .try_iter()
            .filter_map(|status| match status {
                RemasterStatus::EngineProgress { engines_total, .. } => Some(engines_total),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        assert_eq!(max_engines_total, 2);
    }

    #[test]
    fn fallback_wave_skipped_when_cancelled() {
        // Pre-set cancel_flag before remaster_samples runs. Primary wave
        // exits at its first ensure_not_cancelled; fallback guard never runs.
        // Documents the defense-in-depth cancel check at the fallback entry.
        let _guard = env_lock().lock().unwrap();
        let temp_home = tempfile::tempdir().expect("temp home");
        let _home = HomeGuard::set(temp_home.path());
        let work_dir = tempfile::tempdir().expect("temp work");
        let job = sample_job(work_dir.path());

        // Even though caches would trigger fallback, cancellation shortcuts
        // the whole pipeline.
        let low_score_data: Vec<f64> = (0..48_000)
            .map(|i| (i as f64 * 0.017).sin() * 0.5)
            .collect();
        cache_store(
            &job.pcm_sha256,
            "audiosr-v0.1",
            50,
            &SampleResult {
                index: 0,
                data: low_score_data,
                length_frames: 48_000,
                channels: 1,
                sample_rate_hz: 48_000,
                engine_name: "AudioSR".to_string(),
                discovered_loops: None,
            },
            1.0,
        );

        let engine = RemasterEngine::from_test_engines_with_fallback(
            vec![Box::new(DummyEngine {
                name: "AudioSR",
                cache_id: "audiosr-v0.1",
            })],
            vec![Box::new(DummyEngine {
                name: "FLowHigh",
                cache_id: "flowhigh-v0.1",
            })],
        );

        let (progress_tx, _progress_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let cancel_flag = AtomicBool::new(true); // pre-cancelled

        let err = engine
            .remaster_samples(
                vec![job],
                tempfile::tempdir().expect("temp remaster"),
                &progress_tx,
                &result_tx,
                UpscaleMode::CpuOnly,
                &cancel_flag,
                50,
                false,
                false,
            )
            .expect_err("pre-cancelled remaster should return an error");
        assert!(is_cancelled_error(&err));

        let fallback_saw_output = result_rx.try_iter().any(|o| {
            matches!(o,
            RemasterOutput::Candidate(c) | RemasterOutput::Final(c)
                if c.engine_name.contains("FLowHigh"))
        });
        assert!(
            !fallback_saw_output,
            "fallback engine must not run after cancellation",
        );
    }
}
