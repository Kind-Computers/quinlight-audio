# Filament Audio

Filament Audio is a tracker music player and remastering tool for MOD/S3M/XM/IT and
related formats. It plays modules, can remaster their source samples with
optional external AI backends (AudioSR, LavaSR, FLowHigh, AP-BWE), and lets you
A/B the result live during playback.

## Release Scope

- Supported public target: `x86_64-unknown-linux-gnu`
- Supported posture: Linux-first public release, not a cross-platform build
- Default playback/export target: 96 kHz, 32-bit float (64-bit mixed end-to-end)

## What It Does

- Plays tracker formats through vendored libopenmpt with a double-precision mixer
- Opens modules directly from archives (`.zip`, `.7z`, `.rar`, `.tar.*`, `.lha`, `.cab`, `.iso`)
- Replaces samples live during playback so you can compare Original, Reference 48k,
  and AI remasters (AudioSR, LavaSR, FLowHigh, AP-BWE) without restarting the song
- Combines multiple AI engines into a single sample via a per-bin spectral
  consensus on the rotor manifold — bins where the engines agree pass through,
  bins where they disagree (the typical hallucination fingerprint) get
  auto-suppressed
- Exports the live result to FLAC or AAC (256 kbps)
- Supports batch CLI rendering for directories of modules
- Installs as a Linux desktop app (`--install-icon`)

Filament Audio works without AI engines installed. The player, archive support,
reference cleanup path, and export flow remain available even if you never set
up the optional remaster backends.

## Audiophile

Filament Audio's vendored libopenmpt fork is rebuilt for end-to-end double-precision
audio. Every stage from sample interpolation through mixing to output uses 64-bit
floating point — the only quantization in the playback path is the final cast to
f32 at the audio device.

### 64-bit mixer pipeline

The entire mixer bus operates in `double` (`mixsample_t = double`). Volume,
panning, interpolation, and filter feedback all accumulate in 64-bit precision.
Volume ramps use Hermite smoothstep curves (`t²(3−2t)`) instead of linear ramps,
eliminating zipper artifacts on note transitions. The channel filter is a
cascaded 4-pole design — IT-style 2-pole resonant biquad followed by a
Butterworth post-filter — for 24 dB/octave rolloff with no integer truncation
in the coefficient path.

### 48 kHz sample remastering

Each sample in the module can be upscaled to 48 kHz via three methods:

- **AI** (AudioSR / LavaSR / FLowHigh / AP-BWE): neural bandwidth extension
- **48k reference**: deterministic sinc resampling (FFmpeg swresample)
- **Original**: raw sample at native rate (typically 8–22 kHz)

Samples are replaced live during playback. Pattern offset effects (`Oxx`, `SAx`)
are automatically rescaled to match the new sample rate, and portamento effects
are compensated in the engine so pitch slides sound identical regardless of which
sample mode is active.

### Multi-engine consensus

Each enabled AI engine produces its own 48 kHz remaster of every sample.
Filament Audio scores each candidate against the original by Pearson
correlation of magnitude spectra below the source's Nyquist (an engine that
hallucinates even at known frequencies isn't to be trusted), then combines
the engines that pass via a per-bin **Karcher mean on the rotor manifold
ℝ⁺ × S¹**:

- **Magnitude** — geometric mean of the engine magnitudes (Karcher mean on
  ℝ⁺ under multiplication). Smoothly biased toward the quieter engines: the
  rotor-correct successor to softmin, without the per-bin discrete-winner
  ringing of patched-together spectra.
- **Phase** — circular mean of the engine phases (Karcher mean on S¹).
- **Agreement scaling** — the resultant length of the phase rotor sum (0–1)
  multiplies the consensus magnitude. Bins where the engines agree on phase
  pass through at full amplitude; bins where they disagree (the typical
  hallucination fingerprint) are attenuated proportionally.

Below the source's original Nyquist the consensus is then rotor-blended back
toward the source spectrum itself (arithmetic-mean magnitude, shortest-arc
SLERP on phase) so the bottom band stays anchored to the ground truth and
the engines contribute mainly to the band-extension above. Above the source
Nyquist the consensus passes through unchanged.

Why operate on the rotor manifold instead of a Cartesian (complex-linear)
blend: the chord between two phasors of comparable magnitude in ℂ passes
closer to the origin than either endpoint when their phases disagree, so a
linear blend silently attenuates the bin in proportion to phase mismatch —
which the inverse STFT renders as pre-echo and transient smearing.
Operating on the geodesic of (ℝ⁺ × S¹) makes that attenuation explicit
instead of hidden: phase agreement modulates magnitude on purpose, which
both sounds cleaner and is interpretable as a hallucination-rejection
criterion rather than a silent artifact.

### Anisotropic interpolation

Pitch bends (vibrato, portamento, slides) are tracked in full `double` precision
(`PitchT = double`, `FreqT = double`) — no fixed-point period tables or integer
slide accumulators. IT linear slides use `pow(2.0, amount/768.0)` directly.

The resampling filter is a 64-tap polyphase sinc with 65536 phases (16-bit phase
resolution) and an octave-spaced mipmap chain. Each mipmap level tunes Kaiser
window beta independently (β = 14.0 at unity down to β = 8.0 at 128× downsample)
with anisotropic velocity shear coefficients (k_β = 0.65, k_β² = 0.15) that
widen the transition band in proportion to playback speed, keeping the stopband
clean during fast pitch sweeps.

SIMD kernels are compiled for SSE2, AVX, AVX2, and AVX-512 with fully unrolled
accumulator loops — runtime dispatch picks the widest available path.

## Listen

A/B ten tracker modules straight from the repo. The **before** column is the
deterministic render (original samples, no AI); the **after** column is the
same module with samples upscaled by the AI engines. Both clips are 48 kHz
AAC at 256 kbps — downsampled from the engine's 96 kHz default so HTML5
audio in Chrome/Firefox can play them inline. Click to play.

| Module | Format | Before | After |
| --- | --- | --- | --- |
| 2ND_PM | S3M | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/2ND_PM.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/2ND_PM-Filament-Audio-Remastered-96Khz.m4a)** |
| 4mat_-_eternity | XM | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/4mat_-_eternity.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/4mat_-_eternity-Filament-Audio-Remastered-96Khz.m4a)** |
| beyond_the_network | IT | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/beyond_the_network.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/beyond_the_network-Filament-Audio-Remastered-96Khz.m4a)** |
| Caroline | XM | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/Caroline.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/Caroline-Filament-Audio-Remastered-96Khz.m4a)** |
| GroovyUntightFunk | XM | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/GroovyUntightFunk.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/GroovyUntightFunk-Filament-Audio-Remastered-96Khz.m4a)** |
| jt_mind | XM | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/jt_mind.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/jt_mind-Filament-Audio-Remastered-96Khz.m4a)** |
| jt_pools | XM | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/jt_pools.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/jt_pools-Filament-Audio-Remastered-96Khz.m4a)** |
| sweetdre | XM | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/sweetdre.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/sweetdre-Filament-Audio-Remastered-96Khz.m4a)** |
| tiny_tunes | MOD | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/tiny_tunes.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/tiny_tunes-Filament-Audio-Remastered-96Khz.m4a)** |
| znm-wopeace | IT | [listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/rendered/znm-wopeace.m4a) | **[listen](https://github.com/Kind-Computers/filament-audio/raw/main/mods/remastered/znm-wopeace-Filament-Audio-Remastered-96Khz.m4a)** |

Want the full 96 kHz originals? Download
[filament-audio-96khz-bundle.zip](https://github.com/Kind-Computers/filament-audio/releases/download/audio-bundle-v1/filament-audio-96khz-bundle.zip)
(275 MB — all 20 clips at 96 kHz AAC / 256 kbps, organized as
`rendered/` and `remastered/`).

## Build

Filament Audio currently targets Linux `x86_64-unknown-linux-gnu`. The build expects
Rust, a C++ toolchain, SDL2 headers, libarchive headers, and FFmpeg development
libraries.

> **Disk space:** Plan for at least **30 GB free** before installing. The full
> footprint (build artifacts + Python venv + AI model checkpoints) lands around
> 26 GB, with headroom for caches and rendered output.

```bash
sudo apt install build-essential clang mold libsdl2-dev libarchive-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev

cargo build --release
```

## Optional AI Engine Setup

The supported public install path is the checked-in Linux installer:

```bash
./install_prerequisites.sh
```

That script creates `~/.local/share/filament-audio/venv`, installs the pinned Python
package set used by Filament Audio, and runs a simple smoke check at the end.

Supported AI matrix for this release:

- Platform: Linux `x86_64-unknown-linux-gnu`
- Python: `3.12+`
- PyTorch: `2.11.x`
- TorchAudio: `2.11.x`
- TorchVision: `0.26.x`

The GUI shows the same pinned commands if the engines are missing.

## Usage

```bash
# Launch the GUI
filament-audio

# Launch with GPU remastering
filament-audio --upscale-mode gpu

# Render a module to FLAC or AAC at the default 96 kHz target
filament-audio render track.s3m -o track.flac
filament-audio render track.s3m -o track.aac --format aac

# Batch render a directory
filament-audio convert mods -o renders --format flac aac

# Restrict to specific engine(s)
filament-audio convert mods -o renders --engine audiosr --engine lavasr --engine apbwe

# Skip AI remastering (render originals only)
filament-audio convert mods -o renders --no-remaster

# Reference-only cleanup output (no AI, just cleaned 48kHz reference)
filament-audio convert mods -o renders --reference-only --cleanup-preset declick-ar

# Open modules from archives
filament-audio render mods.zip -o track.flac
filament-audio render mods.zip --file track.s3m -o out.flac

# Install .desktop file and icon
filament-audio --install-icon
```

## Legal / Backend Note

AI backend redistribution and branded promotion should still be reviewed
engine-by-engine before any bundled or company-branded release. This
repository documents a supported external-install flow for those backends;
it does not claim that backend weights are bundled or cleared for redistribution.

## License

MIT
