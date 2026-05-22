// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use super::{
    UpsampleEngine, venv_can_import, venv_package_version, venv_python, write_venv_script,
};

const WRAPPER_SCRIPT: &str = r#"
import gc
import json
import os
import sys
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from LavaSR.enhancer.linkwitz_merge import FastLRMerge
from LavaSR.model import LavaEnhance2

device, manifest_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
model = LavaEnhance2("YatharthS/LavaSR", device)
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

def load_conditioning_for_lavasr(file_path, conditioning_rate_hz):
    audio_data, sr = sf.read(file_path, dtype="float64")
    if sr != conditioning_rate_hz:
        raise ValueError(
            f"Expected {conditioning_rate_hz} Hz conditioning WAV, got {sr} for {file_path}"
        )
    if audio_data.ndim != 1:
        raise ValueError(
            f"Expected mono conditioning WAV for LavaSR, got shape {audio_data.shape} for {file_path}"
        )
    audio = torch.from_numpy(audio_data).unsqueeze(0)
    return torchaudio.functional.resample(audio, conditioning_rate_hz, 16000).to(
        model.device, dtype=torch.float32
    )

with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

_ok = _fail = 0
for item in manifest["items"]:
    p = item["conditioning_wav_path"]
    stem = item["stem"]
    conditioning_rate_hz = int(item["conditioning_rate_hz"])
    audio = out = None
    try:
        cutoff = min(float(item["conditioning_lowpass_hz"]), 8000.0)
        model.bwe_model.lr_refiner = FastLRMerge(device=model.device, cutoff=cutoff, transition_bins=1024)
        audio = load_conditioning_for_lavasr(p, conditioning_rate_hz)
        out = model.enhance(audio, denoise=False).cpu().to(torch.float64).numpy().squeeze()
        sf.write(str(Path(outdir) / (stem + ".wav")), out, 48000, subtype='DOUBLE')
        print(f"DONE: {stem}", flush=True)
        _ok += 1
    except Exception as e:
        print(f"FAIL: {stem}: {e}", file=sys.stderr, flush=True)
        _fail += 1
    finally:
        audio = out = None
        _release_memory()
if _fail > 0:
    sys.exit(1)
"#;

pub struct LavaSrEngine {
    script_path: PathBuf,
    cache_id: String,
}

impl LavaSrEngine {
    pub fn detect() -> Option<Box<dyn UpsampleEngine>> {
        if !venv_can_import("LavaSR") {
            return None;
        }
        let script_path = write_venv_script("quinlight_lavasr.py", WRAPPER_SCRIPT).ok()?;
        let version = venv_package_version("LavaSR").unwrap_or_else(|| "unknown".into());
        Some(Box::new(LavaSrEngine {
            script_path,
            cache_id: format!("lavasr-{version}"),
        }))
    }
}

impl UpsampleEngine for LavaSrEngine {
    fn name(&self) -> &str {
        "LavaSR"
    }

    fn cache_id(&self) -> &str {
        &self.cache_id
    }

    fn output_rate(&self) -> u32 {
        48000
    }

    fn max_batch_size(&self) -> usize {
        5
    }

    fn min_duration_secs(&self) -> f64 {
        0.0
    }

    fn spawn_batch(
        &self,
        input_manifest: &Path,
        output_dir: &Path,
        device: &str,
        _ddim_steps: u32,
        cpu_thread_budget: usize,
    ) -> Result<Child, String> {
        let python = venv_python();
        let mut cmd = Command::new(&python);
        super::apply_pytorch_env(&mut cmd, device, cpu_thread_budget);
        cmd.arg(&self.script_path)
            .arg(device)
            .arg(input_manifest)
            .arg(output_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn LavaSR: {e}"))
    }

    fn find_output_wav(&self, output_dir: &Path, stem: &str) -> Result<PathBuf, String> {
        let path = output_dir.join(format!("{stem}.wav"));
        if path.exists() {
            Ok(path)
        } else {
            Err(format!("No output WAV found for '{stem}' in LavaSR output"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lavasr_supports_up_to_16khz() {
        let engine = LavaSrEngine {
            script_path: PathBuf::new(),
            cache_id: "lavasr-test".into(),
        };

        assert!(engine.supports_original_rate(8_000));
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
    fn lavasr_wrapper_uses_manifest_conditioning_rate() {
        assert!(
            WRAPPER_SCRIPT.contains("conditioning_rate_hz = int(item[\"conditioning_rate_hz\"])")
        );
        assert!(WRAPPER_SCRIPT.contains("if sr != conditioning_rate_hz:"));
        assert!(
            WRAPPER_SCRIPT
                .contains("torchaudio.functional.resample(audio, conditioning_rate_hz, 16000)")
        );
    }
}
