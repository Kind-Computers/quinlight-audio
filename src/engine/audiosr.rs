// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use super::{
    UpsampleEngine, package_version_with_python, venv_can_import, venv_python, write_venv_script,
};

/// AudioSR's minimum spectrogram alignment in seconds.
/// Derived from: 512 frames * (480 hop_length / 48000 sample_rate) = 5.12s.
const ALIGNMENT_SECS: f64 = 5.12;

/// Maximum number of samples per AudioSR subprocess invocation.
const MAX_BATCH_SIZE: usize = 5;

const WRAPPER_SCRIPT: &str = r#"
import gc
import json
import os
import sys

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from audiosr.lowpass import lowpass
from audiosr.pipeline import build_model, seed_everything
from audiosr.utils import pad_wav, wav_feature_extraction

device, manifest_path, outdir, ddim_steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(outdir, exist_ok=True)


def _release_memory():
    # Per-item: basicsr registries and CUDA/XPU allocator caches grow across
    # iterations. Drop Python refs and nudge the allocators to release blocks
    # so long batches don't balloon RSS / VRAM.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.from_numpy(x).to(torch.float32)


def read_conditioning_wav_file(filename, conditioning_rate_hz):
    audio_data, sr = sf.read(filename, dtype="float64")
    if sr != conditioning_rate_hz:
        raise ValueError(
            f"Expected {conditioning_rate_hz} Hz conditioning WAV, got {sr} for {filename}"
        )

    if audio_data.ndim != 1:
        raise ValueError(
            f"Expected mono conditioning WAV for AudioSR, got shape {audio_data.shape} for {filename}"
        )

    if conditioning_rate_hz != 48000:
        # AudioSR's model is hard-wired to 48 kHz (STFT hop 480, mel fmax 24 kHz).
        # Kaiser β=13 gives ~130 dB stopband, matching our external FFmpeg SWR SINC.
        audio_data = scipy.signal.resample_poly(
            audio_data,
            up=48000,
            down=conditioning_rate_hz,
            window=("kaiser", 13.0),
        ).astype("float64")
        sr = 48000

    waveform = torch.from_numpy(audio_data[None, :])
    duration = waveform.size(-1) / sr

    if duration > 10.24:
        print(
            "\033[93m {}\033[00m".format(
                "Warning: audio is longer than 10.24 seconds, may degrade the model performance."
            )
        )

    if duration % 5.12 != 0:
        pad_duration = duration + (5.12 - duration % 5.12)
    else:
        pad_duration = duration

    target_frame = int(pad_duration * 100)
    waveform = waveform.numpy()[0, ...]
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, target_length=int(48000 * pad_duration))
    return waveform, target_frame, pad_duration


def lowpass_filtering_prepare_with_cutoff(dl_output, cutoff_freq):
    waveform = dl_output["waveform"]
    sampling_rate = dl_output["sampling_rate"]
    nyquist = sampling_rate * 0.5
    cutoff_freq = max(20.0, min(float(cutoff_freq), nyquist * 0.95))

    order = 8
    ftype = np.random.choice(["butter", "cheby1", "ellip", "bessel"])
    filtered_audio = lowpass(
        waveform.numpy().squeeze(),
        highcut=cutoff_freq,
        fs=sampling_rate,
        order=order,
        _type=ftype,
    )
    filtered_audio = torch.from_numpy(filtered_audio.copy()).unsqueeze(0)

    if waveform.size(-1) <= filtered_audio.size(-1):
        filtered_audio = filtered_audio[..., : waveform.size(-1)]
    else:
        filtered_audio = torch.functional.pad(
            filtered_audio, (0, waveform.size(-1) - filtered_audio.size(-1))
        )

    return {"waveform_lowpass": filtered_audio}


def make_batch_for_super_resolution_with_cutoff(input_file, conditioning_rate_hz, cutoff_hz):
    waveform, target_frame, duration = read_conditioning_wav_file(input_file, conditioning_rate_hz)
    log_mel_spec, stft = wav_feature_extraction(
        waveform.astype("float32") if isinstance(waveform, np.ndarray) else waveform,
        target_frame,
    )

    batch = {
        "waveform": _to_tensor(waveform),
        "stft": _to_tensor(stft),
        "log_mel_spec": _to_tensor(log_mel_spec),
        "sampling_rate": 48000,
    }

    batch.update(lowpass_filtering_prepare_with_cutoff(batch, cutoff_hz))
    waveform_lp = batch["waveform_lowpass"]
    if isinstance(waveform_lp, torch.Tensor):
        waveform_lp = waveform_lp.to(torch.float32)
    elif isinstance(waveform_lp, np.ndarray):
        waveform_lp = waveform_lp.astype("float32")
    lowpass_mel, lowpass_stft = wav_feature_extraction(waveform_lp, target_frame)
    batch["lowpass_mel"] = lowpass_mel

    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(torch.float32).unsqueeze(0)

    return batch, duration


def super_resolution_with_cutoff(
    latent_diffusion,
    input_file,
    conditioning_rate_hz,
    cutoff_hz,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
):
    seed_everything(int(seed))
    batch, duration = make_batch_for_super_resolution_with_cutoff(
        input_file, conditioning_rate_hz, cutoff_hz
    )

    with torch.no_grad():
        waveform = latent_diffusion.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            duration=duration,
        )

    return waveform


with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

audiosr = build_model(model_name="basic", device=device)

for item in manifest["items"]:
    stem = item["stem"]
    input_file = item["conditioning_wav_path"]
    conditioning_rate_hz = int(item["conditioning_rate_hz"])
    cutoff_hz = float(item["conditioning_lowpass_hz"])

    try:
        waveform = super_resolution_with_cutoff(
            audiosr,
            input_file,
            conditioning_rate_hz,
            cutoff_hz,
            seed=42,
            guidance_scale=3.5,
            ddim_steps=ddim_steps,
        )
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        output_path = os.path.join(outdir, f"{stem}.wav")
        sf.write(
            output_path,
            np.squeeze(waveform).astype("float64"),
            samplerate=48000,
            subtype="DOUBLE",
        )
        print(f"DONE: {stem}", flush=True)
    finally:
        waveform = None
        _release_memory()
"#;

pub struct AudioSrEngine {
    python: PathBuf,
    script_path: PathBuf,
    cache_id: String,
}

fn resolve_path_executable(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn audiosr_path_python() -> Option<PathBuf> {
    let audiosr = resolve_path_executable("audiosr")?;
    let first_line = std::fs::read_to_string(&audiosr)
        .ok()?
        .lines()
        .next()?
        .trim()
        .to_string();
    let shebang = first_line.strip_prefix("#!")?;
    let mut parts = shebang.split_whitespace();
    let interpreter = parts.next()?;

    if Path::new(interpreter)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "env")
    {
        let program = parts.find(|part| !part.starts_with('-'))?;
        resolve_path_executable(program).or_else(|| Some(PathBuf::from(program)))
    } else {
        Some(PathBuf::from(interpreter))
    }
}

impl AudioSrEngine {
    /// Detect AudioSR: check our venv first, then PATH.
    pub fn detect() -> Option<Box<dyn UpsampleEngine>> {
        let script_path = write_venv_script("quinlight_audiosr.py", WRAPPER_SCRIPT).ok()?;

        // Check our recommended venv location
        if venv_can_import("audiosr") {
            let python = venv_python();
            if Command::new(&python)
                .arg("-c")
                .arg("import audiosr")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
            {
                let version = package_version_with_python(&python, "audiosr")
                    .unwrap_or_else(|| "unknown".into());
                return Some(Box::new(AudioSrEngine {
                    python,
                    script_path,
                    cache_id: format!("audiosr-{version}"),
                }));
            }
        }

        // Fall back to PATH
        let python = audiosr_path_python()?;
        let available = Command::new(&python)
            .arg("-c")
            .arg("import audiosr")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());

        if available {
            let version =
                package_version_with_python(&python, "audiosr").unwrap_or_else(|| "unknown".into());
            Some(Box::new(AudioSrEngine {
                python,
                script_path,
                cache_id: format!("audiosr-{version}"),
            }))
        } else {
            None
        }
    }
}

impl UpsampleEngine for AudioSrEngine {
    fn name(&self) -> &str {
        "AudioSR"
    }

    fn cache_id(&self) -> &str {
        &self.cache_id
    }

    fn output_rate(&self) -> u32 {
        48000
    }

    fn max_batch_size(&self) -> usize {
        MAX_BATCH_SIZE
    }

    fn min_duration_secs(&self) -> f64 {
        ALIGNMENT_SECS
    }

    fn spawn_batch(
        &self,
        input_manifest: &Path,
        output_dir: &Path,
        device: &str,
        ddim_steps: u32,
        cpu_thread_budget: usize,
    ) -> Result<Child, String> {
        let mut cmd = Command::new(&self.python);
        super::apply_pytorch_env(&mut cmd, device, cpu_thread_budget);
        cmd.arg(&self.script_path)
            .arg(device)
            .arg(input_manifest)
            .arg(output_dir)
            .arg(ddim_steps.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn audiosr wrapper: {e}"))
    }

    fn find_output_wav(&self, output_dir: &Path, stem: &str) -> Result<PathBuf, String> {
        // AudioSR creates a timestamped subdirectory and preserves input filenames,
        // sometimes appending _AudioSR_Processed_48K to the stem.
        let target = format!("{stem}.wav");
        for entry in
            std::fs::read_dir(output_dir).map_err(|e| format!("Failed to read output dir: {e}"))?
        {
            let entry = entry.map_err(|e| format!("Dir entry error: {e}"))?;
            let path = entry.path();
            if path.is_dir() {
                let candidate = path.join(&target);
                if candidate.exists() {
                    return Ok(candidate);
                }
                if let Ok(inner_entries) = std::fs::read_dir(&path) {
                    for inner in inner_entries.flatten() {
                        let inner_path = inner.path();
                        if let Some(name) = inner_path.file_name().and_then(|n| n.to_str())
                            && name.ends_with(".wav")
                            && name.starts_with(stem)
                            && name
                                .as_bytes()
                                .get(stem.len())
                                .is_some_and(|&c| c == b'_' || c == b'.')
                        {
                            return Ok(inner_path);
                        }
                    }
                }
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.ends_with(".wav")
                && name.starts_with(stem)
                && name
                    .as_bytes()
                    .get(stem.len())
                    .is_some_and(|&c| c == b'_' || c == b'.')
            {
                return Ok(path);
            }
        }
        Err(format!(
            "No output WAV found for '{stem}' in output directory"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audiosr_supports_high_original_rates() {
        let engine = AudioSrEngine {
            python: PathBuf::new(),
            script_path: PathBuf::new(),
            cache_id: "audiosr-test".into(),
        };

        assert!(engine.supports_original_rate(16_000));
        assert!(engine.supports_original_rate(22_050));
        assert!(engine.supports_original_rate(44_100));
        assert!(
            !engine.supports_original_rate(48_000),
            "48kHz is already at output rate"
        );
        assert!(!engine.supports_original_rate(96_000), "above output rate");
    }

    #[test]
    fn audiosr_wrapper_does_not_apply_its_own_input_normalization() {
        assert!(!WRAPPER_SCRIPT.contains("normalize_wav("));
        assert!(!WRAPPER_SCRIPT.contains("from audiosr.utils import normalize_wav"));
    }

    #[test]
    fn audiosr_wrapper_accepts_arbitrary_conditioning_rate() {
        // Manifest still drives the rate.
        assert!(
            WRAPPER_SCRIPT.contains("conditioning_rate_hz = int(item[\"conditioning_rate_hz\"])")
        );
        // The 24-kHz-only branch is gone: any non-48 kHz rate is handled uniformly.
        assert!(!WRAPPER_SCRIPT.contains("if conditioning_rate_hz == 24000:"));
        assert!(!WRAPPER_SCRIPT.contains("AudioSR only supports 24 kHz or 48 kHz"));
        // Polyphase resampler with Kaiser β=13 (~130 dB stopband) to match
        // our external FFmpeg SWR quality.
        assert!(WRAPPER_SCRIPT.contains("scipy.signal.resample_poly"));
        assert!(WRAPPER_SCRIPT.contains("up=48000"));
        assert!(WRAPPER_SCRIPT.contains("down=conditioning_rate_hz"));
        assert!(WRAPPER_SCRIPT.contains("(\"kaiser\", 13.0)"));
    }
}
