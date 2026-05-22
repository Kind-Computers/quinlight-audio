// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use super::{UpsampleEngine, venv_has_package, venv_python, write_venv_script};

/// Short commit hash of the pinned AP-BWE upstream, suffixed with a wrapper
/// revision. Bumping this invalidates cached AP-BWE outputs and should be
/// bumped whenever the upstream pin changes OR the wrapper semantics change.
pub(crate) const APBWE_CACHE_TAG: &str = "751710f2-w2";

const WRAPPER_SCRIPT: &str = r#"
import gc
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as aF

device, manifest_path, outdir, apbwe_dir = sys.argv[1:5]
sys.path.insert(0, apbwe_dir)

from env import AttrDict
from datasets.dataset import amp_pha_stft, amp_pha_istft
from models.model import APNet_BWE_Model

_CHECKPOINT_RATES = [8000, 12000, 16000, 24000]
_models = {}


def _pick_checkpoint_rate(orig_rate_hz):
    fits = [r for r in _CHECKPOINT_RATES if r <= orig_rate_hz]
    return max(fits) if fits else _CHECKPOINT_RATES[0]


def _load_model_for_rate(rate_hz):
    if rate_hz in _models:
        return _models[rate_hz]
    rate_k = rate_hz // 1000
    ckpt_file = Path(apbwe_dir) / "checkpoints" / f"g_{rate_k}kto48k"
    config_file = Path(apbwe_dir) / "configs" / f"config_{rate_k}kto48k.json"
    h = AttrDict(json.loads(config_file.read_text()))
    model = APNet_BWE_Model(h).to(device)
    state = torch.load(str(ckpt_file), map_location=device)
    model.load_state_dict(state["generator"])
    model.eval()
    _models[rate_hz] = (model, h)
    return _models[rate_hz]


def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

Path(outdir).mkdir(parents=True, exist_ok=True)

_ok = _fail = 0
for item in manifest["items"]:
    stem = item["stem"]
    audio = audio_nb = amp = pha = amp_g = pha_g = out = None
    try:
        orig_rate = int(item["original_rate_hz"])
        cond_rate = int(item["conditioning_rate_hz"])
        target_rate = _pick_checkpoint_rate(orig_rate)
        model, h = _load_model_for_rate(target_rate)

        audio_data, sr = sf.read(item["conditioning_wav_path"], dtype="float64")
        if sr != cond_rate:
            raise ValueError(
                f"Expected {cond_rate} Hz conditioning WAV, got {sr} for {item['conditioning_wav_path']}"
            )
        if audio_data.ndim != 1:
            raise ValueError(
                f"Expected mono conditioning WAV for AP-BWE, got shape {audio_data.shape}"
            )

        audio = torch.from_numpy(audio_data).to(dtype=torch.float32).unsqueeze(0)
        # Two-step resample matches upstream inference_48k.py: band-limit to the
        # checkpoint's lr_sampling_rate, then resample back up to 48 kHz so the
        # model sees a 48 kHz STFT whose HF bins are ~zero. The (n_fft, hop_size,
        # win_size) from the config are calibrated for 48 kHz framing; feeding
        # them a 16 kHz-rate signal puts every frame on the wrong time grid.
        audio_nb = aF.resample(audio, orig_freq=cond_rate, new_freq=target_rate)
        audio_nb = aF.resample(audio_nb, orig_freq=target_rate, new_freq=48000).to(device)

        with torch.no_grad():
            amp, pha, _ = amp_pha_stft(audio_nb, h.n_fft, h.hop_size, h.win_size)
            amp_g, pha_g, _ = model(amp, pha)
            out_tensor = amp_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size)

        out = out_tensor.squeeze().cpu().to(torch.float64).numpy()
        sf.write(str(Path(outdir) / (stem + ".wav")), out, 48000, subtype="DOUBLE")
        print(f"DONE: {stem}", flush=True)
        _ok += 1
    except Exception as e:
        print(f"FAIL: {stem}: {e}", file=sys.stderr, flush=True)
        _fail += 1
    finally:
        audio = audio_nb = amp = pha = amp_g = pha_g = out = None
        _release_memory()

if _fail > 0:
    sys.exit(1)
"#;

/// Directory where the AP-BWE source checkout lives. Sibling of the shared venv.
pub(crate) fn apbwe_repo_dir() -> PathBuf {
    super::quinlight_data_dir().join("apbwe")
}

fn has_any_48k_checkpoint(ckpt_dir: &Path) -> bool {
    [8u32, 12, 16, 24]
        .iter()
        .any(|rate_k| ckpt_dir.join(format!("g_{rate_k}kto48k")).is_file())
}

pub struct ApBweEngine {
    script_path: PathBuf,
    apbwe_dir: PathBuf,
    cache_id: String,
}

impl ApBweEngine {
    pub fn detect() -> Option<Box<dyn UpsampleEngine>> {
        let apbwe_dir = apbwe_repo_dir();
        // Silent return if the repo isn't cloned — AP-BWE is opt-in.
        if !apbwe_dir.join("models").join("model.py").is_file()
            || !apbwe_dir.join("env.py").is_file()
            || !apbwe_dir.join("datasets").join("dataset.py").is_file()
        {
            return None;
        }
        let ckpt_dir = apbwe_dir.join("checkpoints");
        if !has_any_48k_checkpoint(&ckpt_dir) {
            eprintln!(
                "quinlight: AP-BWE repo at {} has no g_*to48k checkpoint under checkpoints/",
                apbwe_dir.display()
            );
            return None;
        }
        // torch must be importable in the shared venv. Other engines log their
        // own diagnostics if it isn't, so stay quiet here.
        if !venv_has_package("torch") {
            return None;
        }
        let script_path = write_venv_script("quinlight_apbwe.py", WRAPPER_SCRIPT).ok()?;
        Some(Box::new(ApBweEngine {
            script_path,
            apbwe_dir,
            cache_id: format!("apbwe-{APBWE_CACHE_TAG}"),
        }))
    }
}

impl UpsampleEngine for ApBweEngine {
    fn name(&self) -> &str {
        "AP-BWE"
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
            .arg(&self.apbwe_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn AP-BWE: {e}"))
    }

    fn find_output_wav(&self, output_dir: &Path, stem: &str) -> Result<PathBuf, String> {
        let path = output_dir.join(format!("{stem}.wav"));
        if path.exists() {
            Ok(path)
        } else {
            Err(format!("No output WAV found for '{stem}' in AP-BWE output"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub_engine() -> ApBweEngine {
        ApBweEngine {
            script_path: PathBuf::new(),
            apbwe_dir: PathBuf::new(),
            cache_id: "apbwe-test".into(),
        }
    }

    #[test]
    fn apbwe_supports_rates_below_48k() {
        let engine = stub_engine();
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
    fn apbwe_wrapper_picks_checkpoint_by_original_rate() {
        assert!(WRAPPER_SCRIPT.contains("_CHECKPOINT_RATES = [8000, 12000, 16000, 24000]"));
        assert!(WRAPPER_SCRIPT.contains("target_rate = _pick_checkpoint_rate(orig_rate)"));
        assert!(
            WRAPPER_SCRIPT.contains("fits = [r for r in _CHECKPOINT_RATES if r <= orig_rate_hz]")
        );
    }

    #[test]
    fn apbwe_wrapper_writes_double_precision_wav() {
        assert!(WRAPPER_SCRIPT.contains("subtype=\"DOUBLE\""));
        assert!(WRAPPER_SCRIPT.contains("48000"));
    }

    #[test]
    fn apbwe_wrapper_validates_mono_conditioning() {
        assert!(WRAPPER_SCRIPT.contains("Expected mono conditioning WAV for AP-BWE"));
        assert!(WRAPPER_SCRIPT.contains("if audio_data.ndim != 1:"));
    }

    #[test]
    fn apbwe_wrapper_uses_manifest_conditioning_rate() {
        assert!(WRAPPER_SCRIPT.contains("cond_rate = int(item[\"conditioning_rate_hz\"])"));
        assert!(
            WRAPPER_SCRIPT
                .contains("aF.resample(audio, orig_freq=cond_rate, new_freq=target_rate)")
        );
    }

    #[test]
    fn apbwe_wrapper_resamples_back_to_48k_before_stft() {
        // The upstream model is trained with 48 kHz STFT framing (n_fft/hop/win
        // in the config are calibrated for 48 kHz). We must band-limit via
        // target_rate, then resample back up so the STFT grid is correct.
        assert!(
            WRAPPER_SCRIPT.contains("aF.resample(audio_nb, orig_freq=target_rate, new_freq=48000)")
        );
        // Ensure we're feeding the 48 kHz-rate signal to the STFT, not the
        // low-rate one.
        assert!(WRAPPER_SCRIPT.contains("amp_pha_stft(audio_nb"));
    }

    #[test]
    fn apbwe_wrapper_loads_config_per_rate() {
        assert!(WRAPPER_SCRIPT.contains(
            "config_file = Path(apbwe_dir) / \"configs\" / f\"config_{rate_k}kto48k.json\""
        ));
        assert!(WRAPPER_SCRIPT.contains("AttrDict(json.loads(config_file.read_text()))"));
    }

    #[test]
    fn apbwe_wrapper_injects_repo_on_sys_path() {
        assert!(WRAPPER_SCRIPT.contains("sys.path.insert(0, apbwe_dir)"));
        assert!(WRAPPER_SCRIPT.contains("from models.model import APNet_BWE_Model"));
        assert!(
            WRAPPER_SCRIPT.contains("from datasets.dataset import amp_pha_stft, amp_pha_istft")
        );
    }

    #[test]
    fn apbwe_name_matches_upstream_branding() {
        let engine = stub_engine();
        assert_eq!(engine.name(), "AP-BWE");
    }

    #[test]
    fn apbwe_find_output_wav_looks_for_stem_at_top_level() {
        let engine = stub_engine();
        let tmp = tempfile::tempdir().expect("tempdir");
        let stem = "sample_0_L";
        let target = tmp.path().join(format!("{stem}.wav"));
        std::fs::write(&target, b"RIFF").expect("write");
        let got = engine
            .find_output_wav(tmp.path(), stem)
            .expect("should find wav");
        assert_eq!(got, target);
    }

    #[test]
    fn apbwe_find_output_wav_errors_when_missing() {
        let engine = stub_engine();
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(engine.find_output_wav(tmp.path(), "nope").is_err());
    }
}
