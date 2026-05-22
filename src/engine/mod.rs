// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

pub(crate) mod apbwe;
pub(crate) mod audiosr;
pub(crate) mod flowhigh;
pub(crate) mod lavasr;
pub(crate) mod rotor;
pub(crate) mod spectral;

pub use apbwe::ApBweEngine;
pub use audiosr::AudioSrEngine;
pub use flowhigh::FlowHighEngine;
pub use lavasr::LavaSrEngine;
pub(crate) use spectral::fft_cross_correlation;
pub use spectral::spectral_correlation;
pub use spectral::spectral_intersection;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;

/// A backend that can upscale audio samples to a higher sample rate.
///
/// Implementations handle subprocess spawning, GPU selection, and output
/// file naming conventions. The remaster pipeline uses this trait to
/// stay engine-agnostic — only the subprocess interface and file patterns
/// differ between engines.
pub trait UpsampleEngine: Send + Sync {
    /// Human-readable engine name (e.g. "AudioSR", "LavaSR").
    fn name(&self) -> &str;

    /// Stable identifier used in cache keys (e.g. "audiosr-v0.1").
    /// Changing this string invalidates that engine's cached entries.
    fn cache_id(&self) -> &str;

    /// Whether this engine supports a sample with the given original rate.
    /// Samples already at or above the output rate (48kHz) are skipped —
    /// running them through the engine would downscale, not upscale.
    fn supports_original_rate(&self, original_rate_hz: u32) -> bool {
        original_rate_hz < self.output_rate()
    }

    /// Target output sample rate in Hz.
    #[allow(dead_code)]
    fn output_rate(&self) -> u32;

    /// Maximum number of samples per subprocess invocation.
    fn max_batch_size(&self) -> usize;

    /// Minimum input duration in seconds (for padding calculation).
    fn min_duration_secs(&self) -> f64;

    /// Spawn the upsampling subprocess for a batch of files.
    /// `input_manifest` is a JSON manifest describing one or more 48 kHz
    /// conditioning WAVs plus original-rate metadata for the batch.
    /// `output_dir` is where the engine should write output WAVs.
    /// `device` is "cpu" or "cuda".
    /// `ddim_steps` controls quality/speed tradeoff (AudioSR-specific; other engines ignore it).
    /// `cpu_thread_budget` caps PyTorch/OMP/MKL threads for this subprocess so
    /// concurrent engines don't thrash the CPU. Always pass a finite value.
    fn spawn_batch(
        &self,
        input_manifest: &Path,
        output_dir: &Path,
        device: &str,
        ddim_steps: u32,
        cpu_thread_budget: usize,
    ) -> Result<Child, String>;

    /// Find the output WAV for a given input stem in the output directory.
    /// Engines may create subdirectories or rename files — this method
    /// encapsulates the naming convention.
    fn find_output_wav(&self, output_dir: &Path, stem: &str) -> Result<PathBuf, String>;
}

// --- Shared helpers used by all engines ---

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    None,
}

pub(crate) static GPU_VENDOR: OnceLock<GpuVendor> = OnceLock::new();

/// Intel GPU architecture generation, classified by PCI device ID range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum IntelGpuGen {
    Battlemage, // 0xe2xx — Xe2-HPG (BMG), discrete
    Alchemist,  // 0x56xx — Xe-HPG (DG2), Arc A-series
    LunarLake,  // 0x64xx — Xe2-LPG, integrated
    MeteorLake, // 0x7dxx — Xe-LPG, integrated
    Unknown,
}

impl IntelGpuGen {
    fn is_discrete(self) -> bool {
        matches!(self, IntelGpuGen::Battlemage | IntelGpuGen::Alchemist)
    }
}

pub(crate) static INTEL_GPU_GEN: OnceLock<IntelGpuGen> = OnceLock::new();
pub(crate) static XPU_VRAM_MB: OnceLock<u64> = OnceLock::new();

pub(crate) fn detect_gpu() -> GpuVendor {
    if Command::new("nvidia-smi")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
    {
        return GpuVendor::Nvidia;
    }
    if Path::new("/opt/rocm").is_dir() {
        return GpuVendor::Amd;
    }
    if Command::new("xpu-smi")
        .arg("discovery")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
    {
        return GpuVendor::Intel;
    }
    if detect_intel_sysfs() {
        return GpuVendor::Intel;
    }
    GpuVendor::None
}

/// PyTorch device string for the detected GPU vendor.
pub(crate) fn gpu_device_string() -> &'static str {
    match GPU_VENDOR.get_or_init(detect_gpu) {
        GpuVendor::Nvidia | GpuVendor::Amd => "cuda",
        GpuVendor::Intel => "xpu",
        GpuVendor::None => "cpu",
    }
}

/// Check if an Intel GPU is present by scanning DRM sysfs vendor files.
/// Fallback for systems where `xpu-smi` is not installed (e.g. Battlemage
/// with only the Xe kernel driver and Level Zero runtime).
fn detect_intel_sysfs() -> bool {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        // Skip display connectors like "card0-DP-1"
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }
        let vendor_path = entry.path().join("device").join("vendor");
        if let Ok(vendor) = std::fs::read_to_string(&vendor_path)
            && vendor.trim() == "0x8086"
        {
            return true;
        }
    }
    false
}

/// Classify an Intel PCI device ID into the GPU generation.
fn classify_intel_device_id(hex_str: &str) -> IntelGpuGen {
    let id = u16::from_str_radix(hex_str.trim_start_matches("0x"), 16).unwrap_or(0);
    match id {
        0xe200..=0xe2ff => IntelGpuGen::Battlemage,
        0x5600..=0x56ff => IntelGpuGen::Alchemist,
        0x6400..=0x64ff => IntelGpuGen::LunarLake,
        0x7d00..=0x7dff => IntelGpuGen::MeteorLake,
        _ => IntelGpuGen::Unknown,
    }
}

/// Detect the Intel GPU generation by reading PCI device IDs from sysfs.
/// When multiple Intel cards are present (iGPU + dGPU), prefers discrete.
fn detect_intel_gen() -> IntelGpuGen {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return IntelGpuGen::Unknown;
    };
    let mut best = IntelGpuGen::Unknown;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }
        let dev_path = entry.path().join("device");
        let Ok(vendor) = std::fs::read_to_string(dev_path.join("vendor")) else {
            continue;
        };
        if vendor.trim() != "0x8086" {
            continue;
        }
        let Ok(device) = std::fs::read_to_string(dev_path.join("device")) else {
            continue;
        };
        let gpu_gen = classify_intel_device_id(device.trim());
        if gpu_gen.is_discrete() || best == IntelGpuGen::Unknown {
            best = gpu_gen;
        }
        if best.is_discrete() {
            break;
        }
    }
    best
}

/// Return the cached Intel GPU generation (lazy-initialized on first call).
pub(crate) fn intel_gpu_gen() -> IntelGpuGen {
    *INTEL_GPU_GEN.get_or_init(detect_intel_gen)
}

/// Detect Intel XPU local memory in megabytes by parsing the BAR2 aperture
/// from `/sys/class/drm/card*/device/resource`. Returns 0 if unparseable.
pub(crate) fn detect_xpu_vram_mb() -> u64 {
    *XPU_VRAM_MB.get_or_init(detect_xpu_vram_inner)
}

fn detect_xpu_vram_inner() -> u64 {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    // Sort by name so we deterministically pick card0, then card1, etc. —
    // matches PyTorch's default `xpu:0` on most hosts, where Level Zero
    // enumeration follows PCI/sysfs order.
    let mut cards: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card") && !n.contains('-'))
        })
        .collect();
    cards.sort();
    for card in cards {
        let dev_path = card.join("device");
        let Ok(vendor) = std::fs::read_to_string(dev_path.join("vendor")) else {
            continue;
        };
        if vendor.trim() != "0x8086" {
            continue;
        }
        let Ok(resource) = std::fs::read_to_string(dev_path.join("resource")) else {
            continue;
        };
        // BAR2 is the third line (index 2) — the local memory aperture
        if let Some(bar2_line) = resource.lines().nth(2)
            && let Some(size) = parse_pci_bar_size(bar2_line)
        {
            return size / (1024 * 1024);
        }
    }
    0
}

/// Parse a PCI resource line ("start end flags") and return the region size in bytes.
fn parse_pci_bar_size(line: &str) -> Option<u64> {
    let mut parts = line.split_whitespace();
    let start = u64::from_str_radix(parts.next()?.trim_start_matches("0x"), 16).ok()?;
    let end = u64::from_str_radix(parts.next()?.trim_start_matches("0x"), 16).ok()?;
    if end > start {
        Some(end - start + 1)
    } else {
        None
    }
}

/// Query NVIDIA GPU total VRAM in MB via `nvidia-smi`. Returns 0 if unreadable.
///
/// Pins the query to device 0 with `-i 0` because PyTorch defaults to
/// `cuda:0`. `nvidia-smi` honors `CUDA_VISIBLE_DEVICES`, so if the user has
/// remapped devices externally, `-i 0` reports the remapped device — matching
/// what PyTorch will actually use.
pub(crate) fn detect_cuda_vram_mb() -> u64 {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            "0",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    let Ok(output) = output else { return 0 };
    if !output.status.success() {
        return 0;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|l| l.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Query AMD GPU total VRAM in MB via `rocm-smi`. Returns 0 if unreadable.
///
/// Parses rocm-smi's CSV output by locating the "total" (non-"used") memory
/// column from the header row, then reads only the first data row. PyTorch
/// defaults to `cuda:0` (HIP under ROCm), which is the first enumerated
/// device — same row order as `rocm-smi`. If the header lacks an unambiguous
/// total column (which happens when rocm-smi renames or reorders columns
/// across versions), returns 0 — the caller treats that as "trust the user,
/// run all engines."
pub(crate) fn detect_rocm_vram_mb() -> u64 {
    let output = Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--csv"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    let Ok(output) = output else { return 0 };
    if !output.status.success() {
        return 0;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return 0;
    };
    let Some(total_col) = header.split(',').position(|c| {
        let l = c.to_ascii_lowercase();
        l.contains("total") && !l.contains("used")
    }) else {
        return 0;
    };
    let first_bytes = lines
        .find_map(|line| line.split(',').nth(total_col)?.trim().parse::<u64>().ok())
        .unwrap_or(0);
    first_bytes / (1024 * 1024)
}

/// Parse `MemAvailable` from `/proc/meminfo`, in MB. Returns 0 if unreadable.
///
/// `MemAvailable` (kernel 3.14+) is the right number for admission control —
/// it includes reclaimable page cache, unlike `MemFree`. Callers should treat
/// a return of 0 as "trust the user, no RAM cap."
pub(crate) fn detect_available_ram_mb() -> u64 {
    let Ok(content) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    content
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

/// Return total VRAM in MB for the detected GPU vendor, or 0 if unknown/none.
pub(crate) fn detect_gpu_vram_mb() -> u64 {
    match *GPU_VENDOR.get_or_init(detect_gpu) {
        GpuVendor::Nvidia => detect_cuda_vram_mb(),
        GpuVendor::Amd => detect_rocm_vram_mb(),
        GpuVendor::Intel => detect_xpu_vram_mb(),
        GpuVendor::None => 0,
    }
}

/// CPU instruction set level for PyTorch/oneDNN kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CpuIsaLevel {
    /// ATEN_CPU_CAPABILITY: "avx512", "avx2", or "default"
    pub aten: &'static str,
    /// DNNL_MAX_CPU_ISA: "AVX512_CORE", "AVX2", "AVX", "SSE41", or "ALL"
    pub dnnl: &'static str,
}

pub(crate) static CPU_ISA: OnceLock<CpuIsaLevel> = OnceLock::new();

/// Detect the highest CPU ISA level supported by the running hardware.
/// Uses CPUID at runtime so a single binary works on any x86_64 machine.
pub(crate) fn detect_cpu_isa() -> CpuIsaLevel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return CpuIsaLevel {
                aten: "avx512",
                dnnl: "AVX512_CORE",
            };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return CpuIsaLevel {
                aten: "avx2",
                dnnl: "AVX2",
            };
        }
        if std::arch::is_x86_feature_detected!("avx") {
            return CpuIsaLevel {
                aten: "default",
                dnnl: "AVX",
            };
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return CpuIsaLevel {
                aten: "default",
                dnnl: "SSE41",
            };
        }
        CpuIsaLevel {
            aten: "default",
            dnnl: "ALL",
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        CpuIsaLevel {
            aten: "default",
            dnnl: "ALL",
        }
    }
}

/// Apply common PyTorch environment variables to a subprocess Command.
/// Handles GPU-specific vars and CPU ISA-matched dispatch (cached detection).
///
/// `cpu_thread_budget` sets OMP/MKL/OpenBLAS thread counts for this
/// subprocess. Typical value is the full logical-core count so every engine
/// process races for the whole CPU; any throttling policy (nice levels,
/// affinity) is the caller's job, not this function's.
pub(crate) fn apply_pytorch_env(cmd: &mut Command, device: &str, cpu_thread_budget: usize) {
    if device == "cuda" {
        cmd.env("HSA_OVERRIDE_GFX_VERSION", "11.0.0");
        cmd.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
        cmd.env("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True");
    } else if device == "xpu" {
        cmd.env("SYCL_CACHE_PERSISTENT", "1");
        cmd.env("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu");
        if intel_gpu_gen() == IntelGpuGen::Battlemage {
            cmd.env("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1");
            cmd.env("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE");
        }
    }
    let isa = CPU_ISA.get_or_init(detect_cpu_isa);
    cmd.env("ATEN_CPU_CAPABILITY", isa.aten);
    cmd.env("DNNL_MAX_CPU_ISA", isa.dnnl);
    cmd.env("TORCH_CPU_ALLOCATOR", "native");
    cmd.env("TORCHDYNAMO_DISABLE", "1");
    cmd.env("PYTHONWARNINGS", "ignore");
    let threads = cpu_thread_budget.max(1).to_string();
    cmd.env("OMP_NUM_THREADS", &threads);
    cmd.env("MKL_NUM_THREADS", &threads);
    cmd.env("OPENBLAS_NUM_THREADS", &threads);
    // Force MKL and OpenMP to honor *_NUM_THREADS literally per op. Without
    // these, both runtimes dispatch dynamic heuristics that often pick
    // fewer threads than requested (especially for small/medium matmuls and
    // attention heads), which defeats the deliberate-oversubscription
    // experiment. With them set to FALSE, every op is parallelized across
    // all requested threads; the kernel scheduler arbitrates contention.
    cmd.env("MKL_DYNAMIC", "FALSE");
    cmd.env("OMP_DYNAMIC", "FALSE");
}

/// Resolve `~/.local/share/quinlight-audio/`.
pub fn quinlight_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("quinlight-audio")
}

/// Path to our shared venv location.
pub fn venv_dir() -> PathBuf {
    quinlight_data_dir().join("venv")
}

/// Path to the venv's python3 binary.
pub(crate) fn venv_python() -> PathBuf {
    venv_dir().join("bin").join("python3")
}

/// Check if a Python package can be imported in the shared venv.
///
/// Logs a one-line diagnostic to stderr on any failure so that an opaque
/// "no engines found" error message can be traced back to a concrete cause
/// (missing venv, missing package, broken torch install, etc.).
pub(crate) fn venv_can_import(package: &str) -> bool {
    let python = venv_python();
    if !python.exists() {
        eprintln!(
            "quinlight: venv python not found at {} (cannot probe '{package}')",
            python.display()
        );
        return false;
    }
    let output = match Command::new(&python)
        .arg("-c")
        .arg(format!("import {package}"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            eprintln!("quinlight: failed to spawn venv python for 'import {package}': {e}");
            return false;
        }
    };
    if output.status.success() {
        return true;
    }
    // Python tracebacks put the useful line (e.g. "ModuleNotFoundError: ...")
    // at the bottom, not the top. Grab the last non-empty stderr line.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let reason = stderr
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .unwrap_or("(no stderr output)")
        .trim();
    let exit = output
        .status
        .code()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "signal".into());
    eprintln!("quinlight: 'import {package}' failed (exit {exit}): {reason}");
    false
}

/// Silent variant of [`venv_can_import`] — returns whether a package can be
/// imported without logging any diagnostics. Use this when building error
/// messages (where detection has already logged its own failures) or for
/// probing that shouldn't add noise.
pub(crate) fn venv_has_package(package: &str) -> bool {
    let python = venv_python();
    if !python.exists() {
        return false;
    }
    Command::new(&python)
        .arg("-c")
        .arg(format!("import {package}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Query the installed version of a pip package in the shared venv.
/// Returns the version string (e.g. "0.0.7") or None if unavailable.
pub(crate) fn venv_package_version(package: &str) -> Option<String> {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(format!(
            "from importlib.metadata import version; print(version('{package}'))"
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if version.is_empty() {
        None
    } else {
        Some(version)
    }
}

/// Query the installed version of a pip package using a specific python binary.
pub(crate) fn package_version_with_python(python: &Path, package: &str) -> Option<String> {
    let output = Command::new(python)
        .arg("-c")
        .arg(format!(
            "from importlib.metadata import version; print(version('{package}'))"
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if version.is_empty() {
        None
    } else {
        Some(version)
    }
}

/// Write an embedded Python wrapper script to the venv's share/ directory.
/// Returns the path to the written script. Idempotent — overwrites if content changed.
pub(crate) fn write_venv_script(name: &str, content: &str) -> Result<PathBuf, String> {
    let share_dir = venv_dir().join("share");
    std::fs::create_dir_all(&share_dir)
        .map_err(|e| format!("Failed to create {}: {e}", share_dir.display()))?;
    let path = share_dir.join(name);
    // Only write if content changed (avoid unnecessary disk writes)
    if std::fs::read_to_string(&path).ok().as_deref() != Some(content) {
        std::fs::write(&path, content)
            .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
    }
    Ok(path)
}

pub(crate) fn engine_preference_rank(name: &str) -> usize {
    if name.eq_ignore_ascii_case("lavasr") {
        0
    } else if name.eq_ignore_ascii_case("flowhigh") {
        1
    } else if name.eq_ignore_ascii_case("ap-bwe") || name.eq_ignore_ascii_case("apbwe") {
        2
    } else if name.eq_ignore_ascii_case("audiosr") {
        3
    } else {
        4
    }
}

/// Detect all available upsampling engines, in preference order.
/// Probes all engines in parallel (each spawns a Python subprocess to check imports).
pub fn detect_engines() -> Vec<Box<dyn UpsampleEngine>> {
    let (lava, flow, apbwe, audio) = std::thread::scope(|s| {
        let h_lava = s.spawn(LavaSrEngine::detect);
        let h_flow = s.spawn(FlowHighEngine::detect);
        let h_apbwe = s.spawn(ApBweEngine::detect);
        let h_audio = s.spawn(AudioSrEngine::detect);
        (
            h_lava.join().ok().flatten(),
            h_flow.join().ok().flatten(),
            h_apbwe.join().ok().flatten(),
            h_audio.join().ok().flatten(),
        )
    });
    // Prefer LavaSR first, then FLowHigh, then AP-BWE, then AudioSR.
    let mut engines: Vec<Box<dyn UpsampleEngine>> = Vec::new();
    if let Some(e) = lava {
        engines.push(e);
    }
    if let Some(e) = flow {
        engines.push(e);
    }
    if let Some(e) = apbwe {
        engines.push(e);
    }
    if let Some(e) = audio {
        engines.push(e);
    }
    engines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_cpu_isa_returns_valid_values() {
        let isa = detect_cpu_isa();
        assert!(
            matches!(isa.aten, "avx512" | "avx2" | "default"),
            "unexpected ATEN value: {}",
            isa.aten
        );
        assert!(
            matches!(isa.dnnl, "AVX512_CORE" | "AVX2" | "AVX" | "SSE41" | "ALL"),
            "unexpected DNNL value: {}",
            isa.dnnl
        );
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn isa_levels_are_hierarchically_consistent() {
        let isa = detect_cpu_isa();
        // If AVX-512 is detected, AVX2 must also be present (prerequisite)
        if isa.aten == "avx512" {
            assert!(std::arch::is_x86_feature_detected!("avx2"));
        }
        // If AVX-512 DNNL level is set, avx512f must be present
        if isa.dnnl == "AVX512_CORE" {
            assert!(std::arch::is_x86_feature_detected!("avx512f"));
        }
    }

    #[test]
    fn classify_intel_device_ids() {
        assert_eq!(classify_intel_device_id("0xe223"), IntelGpuGen::Battlemage);
        assert_eq!(classify_intel_device_id("0xe200"), IntelGpuGen::Battlemage);
        assert_eq!(classify_intel_device_id("0xe2ff"), IntelGpuGen::Battlemage);
        assert_eq!(classify_intel_device_id("0x56a0"), IntelGpuGen::Alchemist);
        assert_eq!(classify_intel_device_id("0x6400"), IntelGpuGen::LunarLake);
        assert_eq!(classify_intel_device_id("0x7d55"), IntelGpuGen::MeteorLake);
        assert_eq!(classify_intel_device_id("0x0000"), IntelGpuGen::Unknown);
        assert_eq!(classify_intel_device_id("0xffff"), IntelGpuGen::Unknown);
        assert_eq!(classify_intel_device_id("garbage"), IntelGpuGen::Unknown);
    }

    #[test]
    fn intel_gpu_gen_discrete_flag() {
        assert!(IntelGpuGen::Battlemage.is_discrete());
        assert!(IntelGpuGen::Alchemist.is_discrete());
        assert!(!IntelGpuGen::LunarLake.is_discrete());
        assert!(!IntelGpuGen::MeteorLake.is_discrete());
        assert!(!IntelGpuGen::Unknown.is_discrete());
    }

    #[test]
    fn parse_pci_bar_size_valid() {
        // Real BAR2 line from a Battlemage card (32 GB aperture)
        let line = "0x0000001800000000 0x0000001fffffffff 0x000000000014220c";
        let size = parse_pci_bar_size(line).unwrap();
        assert_eq!(size, 0x800000000); // 32 GiB
        assert_eq!(size / (1024 * 1024), 32768); // 32768 MiB
    }

    #[test]
    fn parse_pci_bar_size_zero_region() {
        // Unused BAR — start and end are both 0
        let line = "0x0000000000000000 0x0000000000000000 0x0000000000000000";
        assert!(parse_pci_bar_size(line).is_none());
    }

    #[test]
    fn parse_pci_bar_size_malformed() {
        assert!(parse_pci_bar_size("").is_none());
        assert!(parse_pci_bar_size("not a pci resource line").is_none());
    }
}
