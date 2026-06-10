# Changelog

All notable changes to Quinlight Audio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **True data mip-maps for the Aniso-64 interpolator** (GPU-style
  anisotropic filtering, the design paper's W[q,m]): every sample now carries
  an octave-decimated data pyramid built at load time through a cascaded
  63-tap Kaiser half-band, and the gather blends adjacent mip levels instead
  of swapping kernel cutoffs over full-rate data — 64 taps at level j span
  64·2^j original samples, so heavy pitch-down stays properly bandlimited.
  Decimation is **loop-aware**: loop bodies decimate as periodic signals
  (ping-pong as reflected-periodic, sustain loops included) via per-level
  boundary strips, so key-off tails never bleed across a loop point and loop
  seams stay click-free at any transposition. The kernel table family is now
  fractional (eighth-octave steps over one octave), matched per slice to its
  residual ratio
- **Full sheared-separable anisotropic gather** for the Aniso-64 interpolator
  (design paper Eq. 13/14): the reconstruction footprint now spans up to four
  mip slices widened by `R = 1 + k_r·|μ̇|` (new ctl
  `render.resampler.aniso64_k_r`, default 0.8), each slice's phase taps are
  sheared by `β·(j−μ)`, and the shear velocity is normalized per output sample
  so it no longer varies with the module's tick length (tempo)
- Registration consensus as the default multi-engine merger (dense 1-D
  Lucas–Kanade, pyramidal, flow clamped to ±2 samples) with the winsorized
  robust-mean reducer and the 0.9 usable-score gate on by default
- All module samples normalized to ≥48 kHz (sinc) at interactive load time
- Default stereo separation narrowed to 33% for headphone listening
- GitHub Sponsors link in the README and a `.github/FUNDING.yml`
- README note that the multi-engine AI consensus algorithm is U.S. Patent Pending

### Changed
- Renamed the project from Filament Audio to Quinlight Audio
- **Sample-cache key format**: cache filenames now include loop metadata
  (normal and sustain), source rate/bit depth, and the pipeline version.
  Existing entries are adopted in place (renamed to the new key) when the
  sample's loop shape guarantees the old extraction is identical — non-looped
  samples and ordinary forward loops. Ping-pong, sustain-loop, and tiny
  chip-loop samples re-remaster once (their old extractions could carry the
  reversed-splice / drifted-tail bugs fixed in this release)
- `--engine` / the GUI engine checkboxes are now a hard restriction: engines
  the user does not select no longer run as a hidden second consensus wave
- `convert` interrupted with Ctrl-C now exits with code 130 (was: 1 mid-module
  misreported as a fatal engine failure, or 0 between modules)
- Long-input upsample chunk stitching now uses an equal-gain crossfade
  (equal-power summed the correlated overlap renders to +3 dB at mid-fade)
- HRTF initialization moved off the audio callback (the multi-MB SOFA parse
  used to blow the first callback's deadline and falsely grow the buffer)

### Fixed
- The Aniso-64 loop-start strips disengaged 2^level× too early after each
  loop wrap (`CHN_WRAPPED_LOOP` lifetime was sized for the level-0 kernel),
  letting pre-loop attack content bleed into deep-mip loop passes — up to
  −30 dB at 24× transposition, now below −76 dB (regression-tested)
- Building the mip pyramid for maximum-length samples could throw
  `bad_alloc` out of `PrecomputeLoops` (multi-GiB transient working buffers);
  it now degrades to the level-0 gather instead. Samples too large to fit the
  pyramid in the allocation budget (huge float64 stereo) load without one
  rather than failing to load
- The mixed (sustain+normal) and repeated loop-layout extractions still
  accumulated rounded per-copy lengths (the same drift fixed earlier for
  single loops); both now scale each copy boundary from its native position
- HRTF could never be re-enabled after a failure (the failure latch was
  permanent and the enable command raced its own readiness check); a stale
  dry-delay backlog also survived processor swaps, comb-filtering the dry mix
- Aniso-64 parameter setters (`k_beta`, `k_beta2`, `k_r`) now surface the
  FFI rejection of out-of-range values instead of silently no-opping
- Engine probe subprocesses no longer deadlock on >64 KiB of output and are
  reaped on every error path; `rocminfo` is found at `/opt/rocm/bin` when not
  on `PATH`; the last-resort data dir is per-uid with `0700` permissions
- Burg LPC lattice recursion used a wrong backward-error index, making AR
  click/crackle repair emit clamped full-scale bursts on any noisy signal;
  repairs are now also bounded to ~1.5× the local context level
- `SAx` high-offset rescaling overflowed `u32` for nibbles ≥ 2 at 48 kHz
  targets (silently wrong offsets in release builds, panic in debug)
- Instrument-mode IT modules now resolve `Oxx`/`SAx` rows through the
  instrument note map instead of assuming `instrument == sample + 1`
- Stereo samples with one silent channel could never pass the consensus gate
  (the silent pair scored 0.0 instead of being neutral)
- Looped samples shorter than one FFT frame lost their circular STFT padding
  (a mirror discontinuity exactly at the loop seam)
- Loop seam guard mis-anchored when the sinc master came out a frame short
  (master injected mid-buffer; the true seam left unguarded)
- Ping-pong loop extraction could splice a time-reversed body copy
- Tiled loop boundaries are now scaled per boundary from their native
  positions (the accumulated per-copy rounding put tiny chip loops' key-off
  tail ~0.2 s into wrong content)
- Engine subprocesses are now killed as a process group, waited on with
  cancellation, and `HSA_OVERRIDE_GFX_VERSION` is only set for AMD gfx11
  hardware (it broke ROCm on every other AMD generation)
- Offline render export no longer falsely triggers audio-buffer growth and a
  device bounce; a failed export no longer clobbers an in-flight remaster's
  state machine
- Installer smoke check now detects LavaSR — it was probing the module name `lavasr`, but the package imports as `LavaSR`, so an installed LavaSR was never reported
- Full audit fix pass (2026-06-09): see `docs/AUDIT-2026-06-09.md` for the
  complete findings list; all Critical/High/Medium items addressed, Low items
  addressed or documented as known limitations in-code

## [0.1.0] - 2026-05-10

Initial public release.

### Player
- Vendored libopenmpt fork rebuilt for end-to-end double-precision (`mixsample_t = double`)
- 64-bit mixer pipeline with Hermite smoothstep volume ramps and cascaded 4-pole
  channel filter (IT-style 2-pole resonant biquad + Butterworth post-filter)
- 64-tap polyphase sinc resampler with 65536 phases and an octave-spaced mipmap chain
- Full `double`-precision pitch tracking (`PitchT = double`, `FreqT = double`) for
  vibrato, portamento, and slide effects
- SIMD kernels compiled for SSE2, AVX, AVX2, and AVX-512 with runtime dispatch
- Module loading directly from archives: `.zip`, `.7z`, `.rar`, `.tar.*`, `.lha`,
  `.cab`, `.iso`

### Remastering
- AI sample upscaling via four optional external backends: AudioSR, LavaSR,
  FLowHigh, AP-BWE
- Live A/B between Original, 48 kHz reference (sinc-resampled), and any AI engine
  during playback — no song restart
- Pattern offset effects (`Oxx`, `SAx`) and portamento are auto-rescaled when sample
  rate changes
- Reference-only cleanup pipeline available without AI engines (declick, AR)
- Persistent on-disk sample cache at the platform cache directory

### Rendering
- Export to FLAC or AAC (256 kbps), defaulting to 96 kHz
- Batch CLI rendering for directories of modules with per-engine selection
- HRTF-based binaural rendering option

### Platform
- Linux `x86_64-unknown-linux-gnu` is the supported public target
- Linux desktop integration via `--install-icon` (XDG `.desktop` + 256×256 icon)
- GPU and hybrid CPU/GPU upscaling modes (NVIDIA CUDA, AMD ROCm, Intel XPU)

### Repository
- Public source release at <https://github.com/Kind-Computers/quinlight-audio>
- MIT licensed; AI backends remain external (not bundled)
