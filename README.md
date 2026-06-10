# Quinlight Audio

Quinlight Audio is a tracker music player and remastering tool for MOD/S3M/XM/IT and
related formats. It plays modules, can remaster their source samples with
optional external AI backends (AudioSR, LavaSR, FLowHigh, AP-BWE), and lets you
A/B the result live during playback.

![Quinlight Audio playing "Beyond the Network" with all four AI engines remastered](docs/Screenshot_1.png)

## Release Scope

- Supported public target: `x86_64-unknown-linux-gnu`
- Supported posture: Linux-first public release, not a cross-platform build
- Default playback/export target: 96 kHz, 32-bit float (64-bit mixed end-to-end)

## What It Does

- Plays tracker formats through vendored libopenmpt with a double-precision mixer
- Opens modules directly from archives (`.zip`, `.7z`, `.rar`, `.tar.*`, `.lha`, `.cab`, `.iso`)
- Replaces samples live during playback so you can compare Original, Reference 48k,
  and AI remasters (AudioSR, LavaSR, FLowHigh, AP-BWE) without restarting the song
- Combines multiple AI engines into a single sample via a time-domain
  registration consensus — each engine is aligned to a mathematically-exact
  sinc reference by 1-D dense optical flow, then a per-sample median rejects
  any one engine's hallucinations without the ringing of frequency-domain
  blending (the legacy rotor-manifold spectral consensus stays selectable)
- Exports the live result to FLAC or AAC (256 kbps)
- Supports batch CLI rendering for directories of modules
- Installs as a Linux desktop app (`--install-icon`)

Quinlight Audio works without AI engines installed. The player, archive support,
reference cleanup path, and export flow remain available even if you never set
up the optional remaster backends.

## Audiophile

Quinlight Audio's vendored libopenmpt fork is rebuilt for end-to-end double-precision
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
- **48k reference**: deterministic sinc resampling (FFmpeg swresample) — also
  the alignment master for the registration consensus below
- **Original**: raw sample at native rate (typically 8–22 kHz)

Samples are replaced live during playback. Pattern offset effects (`Oxx`, `SAx`)
are automatically rescaled to match the new sample rate, and portamento effects
are compensated in the engine so pitch slides sound identical regardless of which
sample mode is active.

### Multi-engine consensus

Each enabled AI engine produces its own 48 kHz remaster of every sample. Those
candidates are combined into one sample by a **registration consensus** that
works entirely in the time domain — the default, because it neither rings nor
smears transients the way frequency-domain blending does.

**Registration (default).** The deterministic 48 kHz sinc reference — the
mathematically exact bandlimited upsample — is taken as a *master*. For each AI
engine a dense 1-D Lucas–Kanade optical-flow field is computed from the master
to that engine: the same "register, then reduce" method Quinlight uses on
images, dropped to one dimension. The 2-D structure-tensor solve collapses to
the scalar `u = −Σ(w·Ix·It) / Σ(w·Ix²)` — a per-sample sub-sample shift that
best aligns the engine to the master. Each engine is then linearly warped onto
the master's grid, and the engines are reduced per sample by a **winsorized
robust mean** (the default: each engine is clamped to the per-sample median ±
(k·MAD + floor) before averaging, rejecting a lone spike like the median while
keeping the mean's smoothing where engines agree); plain `median` and `mean`
reducers are selectable via `--reduce`.

- **No ringing.** No FFT and no per-bin magnitude/phase surgery, so none of the
  inverse-STFT pre-echo or transient smearing the spectral path can introduce —
  warp-then-median is phase-coherent by construction.
- **Sample-exact loops.** Every engine is warped onto the sinc master, which
  carries the exact loop length and loop point, and both the LK window and the
  warp wrap circularly across the seam — so looped samples stay aligned to the
  sample.
- **Master guides, gently.** The sinc reference defines the alignment grid and
  the loop timing. In `median` mode it also joins the reduction as a
  self-regulating member (it votes in the source band and is discarded as a
  low outlier in the AI-extended highs); in `mean`/`robust` modes it is
  excluded so its bandlimited (no-HF) content never dilutes the engines'
  band-extension.
- **Robust to a bandlimited master.** A goodness-of-fit gate trusts the flow at
  a sample only when the local model `It ≈ −u·Ix` actually explains the
  engine-vs-master difference (`R² = sxt² / (sxx · Σ w·It²)`). High frequencies
  the master cannot represent therefore cannot drive a spurious shift that would
  mangle the engine's own highs.
- **Strict by default.** Registered mode keeps the per-engine usable-score
  floor (0.9 Pearson correlation against the native-rate original): engines
  scoring below it are dropped for that sample, and a sample with fewer than
  two passing engines keeps its original audio rather than shipping a dubious
  blend. The floor is tunable (`--threshold`), but the default philosophy is
  to leave a sample unremastered before risking a wrong one.

**Spectral (legacy, `--consensus spectral`).** The original path scores each
candidate against the source by Pearson correlation of magnitude spectra below
the source's Nyquist (an engine that hallucinates even at known frequencies
isn't to be trusted), then combines the survivors via a per-bin **Karcher mean
on the rotor manifold ℝ⁺ × S¹**: geometric-mean magnitude (Karcher mean on ℝ⁺,
biased toward the quieter engines), circular-mean phase (Karcher mean on S¹),
and an agreement-scaling term — the resultant length of the phase rotor sum
(0–1) multiplies the magnitude, so bins where the engines disagree on phase (the
hallucination fingerprint) are attenuated. Below the source Nyquist the result
is rotor-blended back toward the source spectrum itself (arithmetic-mean
magnitude, shortest-arc SLERP on phase) so the bottom band stays anchored to
ground truth. Operating on the geodesic of (ℝ⁺ × S¹) makes that attenuation
deliberate instead of the hidden, ringing-inducing attenuation a Cartesian
complex blend produces — but the inverse STFT it still relies on is exactly the
ringing source the registration path removes. It remains available for A/B
comparison and is the automatic fallback when no sinc master is present.

### Anisotropic interpolation

Pitch bends (vibrato, portamento, slides) are tracked in full `double` precision
(`PitchT = double`, `FreqT = double`) — no fixed-point period tables or integer
slide accumulators. IT linear slides use `pow(2.0, amount/768.0)` directly.

The resampling filter is a 64-tap polyphase sinc with 65536 phases (16-bit
phase resolution) over **true per-sample data mip-maps** — every sample
carries an octave-decimated pyramid (GPU texture-chain style, the design
paper's W[q,m]), built at load time through a cascaded 63-tap Kaiser
half-band. Reading mip level j makes the 64 taps span 64·2^j original
samples, so heavy pitch-down stays properly bandlimited at any ratio. The
decimation is **loop-aware**: loop bodies decimate as periodic signals
(ping-pong as reflected-periodic, sustain loops included) via per-level
boundary strips, so nothing past a loop point ever bleeds into the loop and
seams stay click-free at any transposition. Each slice samples its mip with a
kernel matched to the residual ratio (a fractional one-octave Kaiser family,
β = 14.0 at unity to β = 11.0 near the octave edge).

On top of the data pyramid sits the full **sheared-separable anisotropic
gather** from the design paper (Eq. 13/14): the reconstruction footprint
spans up to four mip slices around the continuous level μ, widened by
`R = 1 + k_r·|μ̇|` when the pitch is moving (`render.resampler.aniso64_k_r`,
default 0.8), with stretched-tent slice weights that reduce exactly to the
classic trilinear blend at steady pitch. Each slice's phase taps are sheared
by `β·(j−μ)` where `β = k_β·İ + k_β²·Ï` (`aniso64_k_beta` = 0.65,
`aniso64_k_beta2` = 0.15) — per-output-sample, tempo-invariant derivatives of
the playback increment, so the shear strength no longer depends on the
module's tick length.

Full derivation and design notes:
[audio_anisotropic_filter_v2.pdf](docs/audio_anisotropic_filter_v2.pdf).
(The PDF's §12 "Connection to real engines" describes the pre-Aniso-64
16-tap engine; the shipped default is the 64-tap gather above.)

SIMD kernels are compiled for SSE2, AVX, AVX2, and AVX-512 with fully unrolled
accumulator loops — runtime dispatch picks the widest available path.

## Listen

A/B ten tracker modules straight from the repo. The **before** column is the
deterministic render (original samples, no AI); the **after** column is the
same module with samples upscaled by the AI engines. Both clips are 48 kHz
MP3 at 320 kbps — downsampled from the engine's 96 kHz default and served
via GitHub Pages so every browser plays them inline. Click to listen.

| Module | Format | Before | After |
| --- | --- | --- | --- |
| 2ND_PM | S3M | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/2ND_PM.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/2ND_PM-Quinlight-Audio-Remastered-48Khz.mp3)** |
| 4mat_-_eternity | XM | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/4mat_-_eternity.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/4mat_-_eternity-Quinlight-Audio-Remastered-48Khz.mp3)** |
| beyond_the_network | IT | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/beyond_the_network.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/beyond_the_network-Quinlight-Audio-Remastered-48Khz.mp3)** |
| Caroline | XM | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/Caroline.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/Caroline-Quinlight-Audio-Remastered-48Khz.mp3)** |
| GroovyUntightFunk | XM | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/GroovyUntightFunk.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/GroovyUntightFunk-Quinlight-Audio-Remastered-48Khz.mp3)** |
| jt_mind | XM | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/jt_mind.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/jt_mind-Quinlight-Audio-Remastered-48Khz.mp3)** |
| jt_pools | XM | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/jt_pools.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/jt_pools-Quinlight-Audio-Remastered-48Khz.mp3)** |
| sweetdre | XM | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/sweetdre.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/sweetdre-Quinlight-Audio-Remastered-48Khz.mp3)** |
| tiny_tunes | MOD | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/tiny_tunes.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/tiny_tunes-Quinlight-Audio-Remastered-48Khz.mp3)** |
| znm-wopeace | IT | [listen](https://kind-computers.github.io/quinlight-audio/mods/rendered/znm-wopeace.mp3) | **[listen](https://kind-computers.github.io/quinlight-audio/mods/remastered/znm-wopeace-Quinlight-Audio-Remastered-48Khz.mp3)** |

### 96 kHz AAC originals

Same 20 clips at the engine's native 96 kHz / AAC 256 kbps, also served via
GitHub Pages. Many browser AAC decoders accept 96 kHz files but resample to
the system output rate at playback time — what you hear may not be the full
96 kHz, but the bytes your browser fetched are.

| Module | Before | After |
| --- | --- | --- |
| 2ND_PM | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/2ND_PM.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/2ND_PM-Quinlight-Audio-Remastered-96Khz.m4a)** |
| 4mat_-_eternity | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/4mat_-_eternity.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/4mat_-_eternity-Quinlight-Audio-Remastered-96Khz.m4a)** |
| beyond_the_network | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/beyond_the_network.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/beyond_the_network-Quinlight-Audio-Remastered-96Khz.m4a)** |
| Caroline | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/Caroline.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/Caroline-Quinlight-Audio-Remastered-96Khz.m4a)** |
| GroovyUntightFunk | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/GroovyUntightFunk.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/GroovyUntightFunk-Quinlight-Audio-Remastered-96Khz.m4a)** |
| jt_mind | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/jt_mind.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/jt_mind-Quinlight-Audio-Remastered-96Khz.m4a)** |
| jt_pools | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/jt_pools.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/jt_pools-Quinlight-Audio-Remastered-96Khz.m4a)** |
| sweetdre | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/sweetdre.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/sweetdre-Quinlight-Audio-Remastered-96Khz.m4a)** |
| tiny_tunes | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/tiny_tunes.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/tiny_tunes-Quinlight-Audio-Remastered-96Khz.m4a)** |
| znm-wopeace | [listen](https://kind-computers.github.io/quinlight-audio/96khz/rendered/znm-wopeace.m4a) | **[listen](https://kind-computers.github.io/quinlight-audio/96khz/remastered/znm-wopeace-Quinlight-Audio-Remastered-96Khz.m4a)** |

Prefer the bundled download? Grab
[quinlight-audio-96khz-bundle.zip](https://github.com/Kind-Computers/quinlight-audio/releases/download/audio-bundle-v1/quinlight-audio-96khz-bundle.zip)
(275 MB — same 20 clips organized as `rendered/` and `remastered/`).

## Build

Quinlight Audio currently targets Linux `x86_64-unknown-linux-gnu`. The build expects
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

That script creates `~/.local/share/quinlight-audio/venv`, installs the pinned Python
package set used by Quinlight Audio, and runs a simple smoke check at the end.

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
quinlight-audio

# Launch with GPU remastering
quinlight-audio --upscale-mode gpu

# Render a module to FLAC or AAC at the default 96 kHz target
quinlight-audio render track.s3m -o track.flac
quinlight-audio render track.s3m -o track.aac --format aac

# Batch render a directory
quinlight-audio convert mods -o renders --format flac aac

# Restrict to specific engine(s)
quinlight-audio convert mods -o renders --engine audiosr --engine lavasr --engine apbwe

# Skip AI remastering (render originals only)
quinlight-audio convert mods -o renders --no-remaster

# Reference-only cleanup output (no AI, just cleaned 48kHz reference)
quinlight-audio convert mods -o renders --reference-only --cleanup-preset declick-ar

# Open modules from archives
quinlight-audio render mods.zip -o track.flac
quinlight-audio render mods.zip --file track.s3m -o out.flac

# Install .desktop file and icon
quinlight-audio --install-icon
```

## Sponsor

Quinlight Audio is built by Kind Computers, LLC. If it's useful to you and you'd
like to help fund continued development, you can sponsor the project on GitHub:

[**❤ Sponsor Quinlight Audio on GitHub**](https://github.com/sponsors/Kind-Computers)

## Legal / Backend Note

AI backend redistribution and branded promotion should still be reviewed
engine-by-engine before any bundled or company-branded release. This
repository documents a supported external-install flow for those backends;
it does not claim that backend weights are bundled or cleared for redistribution.

**Patent pending.** Quinlight Audio's multi-engine AI consensus algorithm — the
per-bin Karcher-mean spectral consensus on the rotor manifold described under
[Multi-engine consensus](#multi-engine-consensus) — is the subject of a pending
U.S. patent application.

## License

MIT
