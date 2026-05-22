// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use super::{UpsampleEngine, venv_package_version, venv_python, write_venv_script};

const WRAPPER_SCRIPT: &str = r#"
import gc, sys, os, json, librosa, numpy as np, scipy, torch, soundfile as sf
from pathlib import Path

device_arg = sys.argv[1]


def _release_memory():
    # Per-item: basicsr registries and CUDA/XPU allocator caches grow across
    # iterations. Drop Python refs and nudge the allocators to release blocks
    # so long batches don't balloon RSS / VRAM.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

# FLowHigh hardcodes .cuda() throughout (init_bigvgan, from_local, generate,
# cfm_superresolution, postprocessing). On non-CUDA devices, redirect .cuda()
# to .to(device_arg) so tensors land on the correct device (CPU identity, XPU
# transfer, etc.) instead of crashing on a missing CUDA runtime.
if device_arg != "cuda":
    _target_dev = device_arg
    torch.Tensor.cuda = lambda self, *a, _d=_target_dev, **kw: self.to(_d)
    torch.nn.Module.cuda = lambda self, *a, _d=_target_dev, **kw: self.to(_d)

# Neutralize beartype — FLowHigh decorates RotaryEmbedding.forward() with
# @beartype using a Union[int, Tensor] hint that newer beartype rejects.
# beartype is a runtime type-checker only; disabling it is safe.
import beartype
beartype.beartype = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)

# Monkey-patch FLowHigh to avoid hardcoded .cuda() calls that fail on AMD ROCm.
# The upstream library has .cuda() and map_location="cuda" sprinkled throughout
# init_bigvgan, from_local, and generate. We patch all three.
import flowhigh.models.bigvgan.init_vocoder as _iv
import flowhigh.models.melvoco as _mv
from flowhigh.models.bigvgan.models import BigVGAN
from flowhigh.models.bigvgan.env import AttrDict

def _patched_init_bigvgan(config, checkpoint, vocoder_freeze=False):
    with open(config) as f:
        h = AttrDict(json.load(f))
    vocoder = BigVGAN(h)
    checkpoint_dict = torch.load(checkpoint, map_location="cpu")
    vocoder.load_state_dict(checkpoint_dict["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    if vocoder_freeze:
        for param in vocoder.parameters():
            param.requires_grad = False
    return vocoder

_iv.init_bigvgan = _patched_init_bigvgan
_mv.init_bigvgan = _patched_init_bigvgan

from flowhigh import FlowHighSR
from flowhigh.models import FLowHigh, MelVoco
from flowhigh.postprocessing import PostProcessing
from torchaudio.transforms import Spectrogram, InverseSpectrogram

# Patch PostProcessing to avoid hardcoded .cuda(rank) — place on correct device
def _patched_pp_init(self, rank):
    self.stft = Spectrogram(2048, hop_length=480, win_length=2048, power=None, pad_mode='constant')
    self.istft = InverseSpectrogram(2048, hop_length=480, win_length=2048, pad_mode='constant')
    if device_arg != "cpu":
        self.stft = self.stft.to(device_arg)
        self.istft = self.istft.to(device_arg)
PostProcessing.__init__ = _patched_pp_init

@classmethod
def _patched_from_local(cls, ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    voc = MelVoco(
        vocoder_config=ckpt_dir / "bigvgan_48khz_256band.json",
        vocoder_path=ckpt_dir / "bigvgan_48khz_256band.pt",
    )
    SR_generator = FLowHigh(dim_in=voc.n_mels, audio_enc_dec=voc, depth=2)
    SR_generator = SR_generator.to(device).eval()
    cfm_wrapper = cls(flowhigh=SR_generator)
    model_checkpoint = torch.load(ckpt_dir / "FLowHigh_basic_400k.pt", map_location=device)
    cfm_wrapper.load_state_dict(model_checkpoint["model"])
    cfm_wrapper = cfm_wrapper.to(device).eval()
    return cfm_wrapper

FlowHighSR.from_local = _patched_from_local

@torch.no_grad()
def _patched_generate(self, audio, sr, target_sampling_rate=48000, timestep=1):
    if len(audio.shape) == 2:
        audio = audio.squeeze(0)

    if audio.max() > 1:
        audio = audio / 32768.0

    if self.upsampling_method == 'scipy':
        cond = scipy.signal.resample_poly(audio, target_sampling_rate, sr)
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond).unsqueeze(0)
        cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]

    elif self.upsampling_method == 'librosa':
        cond = librosa.resample(audio, sr, target_sampling_rate, res_type='soxr_hq')
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond).unsqueeze(0)
        cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]

    if isinstance(cond, np.ndarray):
        cond = torch.from_numpy(cond)

    cond = cond.to(self.device, dtype=torch.float32)

    if self.cfm_method == 'basic_cfm':
        HR_audio = self.sample(cond=cond, time_steps=timestep, cfm_method=self.cfm_method)
    elif self.cfm_method == 'independent_cfm_adaptive':
        HR_audio = self.sample(
            cond=cond,
            time_steps=timestep,
            cfm_method=self.cfm_method,
            std_2=1.,
        )
    elif self.cfm_method == 'independent_cfm_constant':
        HR_audio = self.sample(cond=cond, time_steps=timestep, cfm_method=self.cfm_method)
    elif self.cfm_method == 'independent_cfm_mix':
        HR_audio = self.sample(cond=cond, time_steps=timestep, cfm_method=self.cfm_method)

    HR_audio = HR_audio.squeeze(1) # [1, T]
    HR_audio_pp = self.postproc.post_processing(HR_audio, cond, cond.size(-1)) # [1, T]
    return HR_audio_pp

FlowHighSR.generate = _patched_generate

device = device_arg
manifest_path, outdir = sys.argv[2], sys.argv[3]

model = FlowHighSR.from_pretrained(device="cpu")
if device != "cpu":
    model = model.to(device)

with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

_ok = _fail = 0
for item in manifest["items"]:
    p = item["conditioning_wav_path"]
    stem = item["stem"]
    conditioning_rate_hz = int(item["conditioning_rate_hz"])
    audio = wav = out = out_np = None
    try:
        audio, sr = sf.read(p, dtype="float64")
        if sr != conditioning_rate_hz:
            raise ValueError(
                f"Expected {conditioning_rate_hz} Hz conditioning WAV, got {sr} for {p}"
            )
        if audio.ndim != 1:
            raise ValueError(
                f"Expected mono conditioning WAV for FLowHigh, got shape {audio.shape} for {p}"
            )
        wav = torch.from_numpy(audio).unsqueeze(0)
        out = model.generate(wav, sr, 48000)
        out_np = out.cpu().to(torch.float64).squeeze(0).numpy()
        if out_np.ndim == 2:
            out_np = out_np.T
        sf.write(str(Path(outdir) / (stem + ".wav")), out_np, 48000, subtype='DOUBLE')
        print(f"DONE: {stem}", flush=True)
        _ok += 1
    except Exception as e:
        print(f"FAIL: {stem}: {e}", file=sys.stderr, flush=True)
        _fail += 1
    finally:
        audio = wav = out = out_np = None
        _release_memory()
if _ok == 0 and _fail > 0:
    sys.exit(1)
"#;

pub struct FlowHighEngine {
    script_path: PathBuf,
    cache_id: String,
}

impl FlowHighEngine {
    pub fn detect() -> Option<Box<dyn UpsampleEngine>> {
        // FLowHigh's @beartype decorator rejects its own Union[int, Tensor]
        // hint in newer beartype versions.  Neutralise beartype before
        // importing so detection succeeds when the package is installed.
        let python = venv_python();
        if !python.exists() {
            return None;
        }
        let output = std::process::Command::new(&python)
            .arg("-c")
            .arg("import beartype; beartype.beartype = lambda f: f; import flowhigh")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output();
        match output {
            Ok(o) if o.status.success() => {}
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                let reason = stderr
                    .lines()
                    .rfind(|l| !l.trim().is_empty())
                    .unwrap_or("(no stderr)")
                    .trim();
                let exit = o
                    .status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into());
                eprintln!("quinlight: 'import flowhigh' failed (exit {exit}): {reason}");
                return None;
            }
            Err(_) => return None,
        }
        let script_path = write_venv_script("quinlight_flowhigh.py", WRAPPER_SCRIPT).ok()?;
        let version = venv_package_version("flowhigh").unwrap_or_else(|| "unknown".into());
        Some(Box::new(FlowHighEngine {
            script_path,
            cache_id: format!("flowhigh-{version}"),
        }))
    }
}

impl UpsampleEngine for FlowHighEngine {
    fn name(&self) -> &str {
        "FLowHigh"
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
        if device == "cpu" {
            // Hide GPU from FLowHigh on CPU to prevent HIP/ROCm kernel errors
            // (the library has hardcoded .cuda() calls that poison the process)
            cmd.env("HIP_VISIBLE_DEVICES", "-1");
            cmd.env("CUDA_VISIBLE_DEVICES", "");
        }
        cmd.arg(&self.script_path)
            .arg(device)
            .arg(input_manifest)
            .arg(output_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn FLowHigh: {e}"))
    }

    fn find_output_wav(&self, output_dir: &Path, stem: &str) -> Result<PathBuf, String> {
        let path = output_dir.join(format!("{stem}.wav"));
        if path.exists() {
            Ok(path)
        } else {
            Err(format!(
                "No output WAV found for '{stem}' in FLowHigh output"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flowhigh_supports_high_original_rates() {
        let engine = FlowHighEngine {
            script_path: PathBuf::new(),
            cache_id: "flowhigh-test".into(),
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
    fn flowhigh_wrapper_does_not_renormalize_conditioning_audio() {
        assert!(!WRAPPER_SCRIPT.contains("np.max(np.abs(cond))"));
        assert!(WRAPPER_SCRIPT.contains("FlowHighSR.generate = _patched_generate"));
    }
}
