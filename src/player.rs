// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use crossbeam_channel::{Receiver, Sender};
use sdl3::audio::{AudioCallback, AudioFormat, AudioSpec, AudioStream, AudioStreamWithCallback};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::openmpt::{
    AgcProfile, DEFAULT_AGC_ENABLED, DEFAULT_AGC_PROFILE, DEFAULT_INTERPOLATION_FILTER_LENGTH,
    DEFAULT_STEREO_SEPARATION_PERCENT, Module,
};

const DEFAULT_PLAYBACK_RATE: u32 = 48_000;
const PREFERRED_PLAYBACK_RATES: &[u32] = &[96_000, 88_200, 48_000, 44_100];
/// Frames rendered per SDL3 callback chunk. The push-based callback loops in
/// slices this size so each `render_audio_f64` / HRTF `process` call stays
/// within the HRTF processor's input ceiling regardless of how large a block
/// SDL3 requests at once.
const CALLBACK_CHUNK_FRAMES: usize = 1024;
const OSCILLOSCOPE_BUFFER_SAMPLES: usize = 8192;

#[derive(Debug, Clone)]
pub enum PlayerCommand {
    Play,
    Pause,
    Stop,
    Seek(f64),
    SetInterpolation(i32),
    SetStereoSeparation(i32),
    SetAgcEnabled(bool),
    SetVolume(f64),
    SetHrtfMix(i32),
    PlaySample { data: Vec<f64>, rate_ratio: f64 },
}

pub struct PreparedModuleLoad {
    file_data: Vec<u8>,
    module: Module,
    file_size_bytes: u64,
    sample_stash: Vec<crate::remaster::NormalizedSample>,
}

impl std::fmt::Debug for PreparedModuleLoad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedModuleLoad")
            .field("file_size_bytes", &self.file_size_bytes)
            .finish()
    }
}

impl PreparedModuleLoad {
    pub fn file_size_bytes(&self) -> u64 {
        self.file_size_bytes
    }

    pub fn clone_file_data(&self) -> Vec<u8> {
        self.file_data.clone()
    }
}

pub fn prepare_module_load_from_path(path: &Path) -> Result<PreparedModuleLoad, String> {
    let file_data = crate::archive::read_module_file(path)?;
    prepare_module_load_from_bytes(file_data)
}

pub fn prepare_module_load_from_bytes(file_data: Vec<u8>) -> Result<PreparedModuleLoad, String> {
    let file_size_bytes = file_data.len() as u64;
    if file_size_bytes > crate::archive::MAX_MODULE_BYTES {
        return Err(format!(
            "Refusing to load module bytes: {} bytes exceeds the {} byte safety limit",
            file_data.len(),
            crate::archive::MAX_MODULE_BYTES
        ));
    }

    let mut module = Module::from_memory(&file_data).map_err(|e| format!("Load failed: {e}"))?;
    // Normalize every sub-48 kHz sample up to 48 kHz (band-limited sinc) at load
    // time so that, during an interactive remaster, swapping an AI-upscaled
    // (48 kHz) sample into the live module is a 48k -> 48k replacement with no
    // rate change — eliminating the per-voice pitch/timing tick. The pristine
    // native samples are stashed and later fed to the AI unchanged. This runs
    // here (off the UI thread and off the audio lock) because `prepare` is
    // dispatched via `Task::perform`. Skippable via QUINLIGHT_NO_LOAD_NORMALIZE
    // for A/B debugging (the GUI then falls back to reading native samples).
    let sample_stash = if std::env::var_os("QUINLIGHT_NO_LOAD_NORMALIZE").is_none() {
        crate::remaster::normalize_samples_to_48k(&mut module)
    } else {
        Vec::new()
    };
    Ok(PreparedModuleLoad {
        file_data,
        module,
        file_size_bytes,
        sample_stash,
    })
}

const MAX_KEYJAZZ_VOICES: usize = 8;

struct KeyjazzVoice {
    data: Vec<f64>,
    position: f64,
    rate_ratio: f64,
    active: bool,
}

impl KeyjazzVoice {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            position: 0.0,
            rate_ratio: 1.0,
            active: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PlayerState {
    pub status: PlaybackStatus,
    pub position_seconds: f64,
    pub duration_seconds: f64,
    pub current_order: i32,
    pub current_row: i32,
    pub current_pattern: i32,
    pub channel_vu: Vec<(f64, f64)>,
    pub active_samples: Vec<i32>,
    pub error: Option<String>,
    pub bpm: f64,
    pub speed: i32,
    pub load_generation: u64,
    /// Set when background HRTF initialization failed; the GUI unchecks its
    /// HRTF toggle and surfaces `error` instead of silently playing dry.
    pub hrtf_failed: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum PlaybackStatus {
    #[default]
    Stopped,
    Playing,
    Paused,
}

struct PlayerInner {
    module: Option<Module>,
    status: PlaybackStatus,
    file_data: Option<Vec<u8>>,
    volume: f64,
    fade_target: f64,
    keyjazz_voices: Vec<KeyjazzVoice>,
    load_generation: u64,
    interpolation_filter: i32,
    stereo_separation: i32,
    agc_enabled: bool,
    agc_profile: AgcProfile,
    hrtf_enabled: bool,
    hrtf_mix: i32,
    hrtf_processor: Option<crate::hrtf::HrtfProcessor>,
    /// Sample rate the resident processor was built for; a device reopen at a
    /// different rate triggers a rebuild.
    hrtf_rate: u32,
    /// True while a background `spawn_hrtf_init` thread is running.
    hrtf_init_in_flight: bool,
    hrtf_dry_buf: Vec<f64>,
    /// FIFO delaying the dry path by the wet path's startup latency so the
    /// wet/dry mix stays phase-aligned for non-partition-multiple buffers.
    hrtf_dry_delay: Vec<f64>,
    /// Pristine native (< 48 kHz) samples captured at load, each paired with the
    /// 48 kHz master now resident in the module. The GUI remaster sources its
    /// originals from here so the AI sees native input while live swaps stay at
    /// 48 kHz. Replaced on every load; empty when normalization is disabled.
    sample_stash: Vec<crate::remaster::NormalizedSample>,
}

impl PlayerInner {
    fn new() -> Self {
        let voices = (0..MAX_KEYJAZZ_VOICES)
            .map(|_| KeyjazzVoice::new())
            .collect();
        Self {
            module: None,
            status: PlaybackStatus::Stopped,
            file_data: None,
            volume: 1.0,
            fade_target: 1.0,
            keyjazz_voices: voices,
            load_generation: 0,
            interpolation_filter: DEFAULT_INTERPOLATION_FILTER_LENGTH,
            stereo_separation: DEFAULT_STEREO_SEPARATION_PERCENT,
            agc_enabled: DEFAULT_AGC_ENABLED,
            agc_profile: DEFAULT_AGC_PROFILE,
            hrtf_enabled: true,
            hrtf_mix: 33,
            hrtf_processor: None,
            hrtf_rate: 0,
            hrtf_init_in_flight: false,
            hrtf_dry_buf: Vec::new(),
            hrtf_dry_delay: Vec::new(),
            sample_stash: Vec::new(),
        }
    }
}

fn install_prepared_load(
    player: &mut PlayerInner,
    state: &mut PlayerState,
    prepared: PreparedModuleLoad,
) {
    // The old module's keyjazz voices must not ring into the new module.
    for voice in &mut player.keyjazz_voices {
        voice.active = false;
        voice.data = Vec::new();
    }
    let PreparedModuleLoad {
        file_data,
        mut module,
        file_size_bytes: _,
        sample_stash,
    } = prepared;
    module.set_repeat_count(-1);
    apply_module_processing_settings(
        &mut module,
        player.stereo_separation,
        player.interpolation_filter,
        player.agc_enabled,
        player.agc_profile,
    );
    player.file_data = Some(file_data);
    player.module = Some(module);
    player.sample_stash = sample_stash;
    player.status = PlaybackStatus::Playing;
    player.load_generation += 1;

    state.error = None;
    state.status = PlaybackStatus::Playing;
    state.position_seconds = 0.0;
    state.current_order = 0;
    state.current_row = 0;
    state.current_pattern = 0;
    state.duration_seconds = player
        .module
        .as_ref()
        .map(|module| module.duration_seconds())
        .unwrap_or(0.0);
    state.channel_vu.clear();
    state.active_samples.clear();
    state.bpm = 0.0;
    state.speed = 0;
    state.load_generation = player.load_generation;
}

fn apply_module_processing_settings(
    module: &mut Module,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    agc_profile: AgcProfile,
) {
    module.apply_quinlight_processing_settings(
        stereo_separation,
        interpolation_filter,
        agc_enabled,
    );
    module.set_agc_profile(agc_profile);
}

fn waveform_frame_slice(output: &[f64], rendered_frames: usize) -> &[f64] {
    let rendered_samples = rendered_frames.saturating_mul(2).min(output.len());
    let keep = rendered_samples.min(OSCILLOSCOPE_BUFFER_SAMPLES);
    let even_keep = keep - (keep % 2);
    &output[..even_keep]
}

fn sync_waveform_buffer(
    waveform: &mut Vec<f64>,
    status: &PlaybackStatus,
    output: &[f64],
    rendered_frames: usize,
) {
    match status {
        PlaybackStatus::Playing => {
            waveform.clear();
            waveform.extend_from_slice(waveform_frame_slice(output, rendered_frames));
        }
        PlaybackStatus::Paused => {}
        PlaybackStatus::Stopped => waveform.clear(),
    }
}

/// Process pending commands from the GUI thread.
fn process_audio_commands(inner: &Mutex<PlayerInner>, command_rx: &Receiver<PlayerCommand>) {
    while let Ok(cmd) = command_rx.try_recv() {
        let mut player = inner.lock().unwrap();
        match cmd {
            PlayerCommand::Play => {
                if player.module.is_some() {
                    player.status = PlaybackStatus::Playing;
                }
            }
            PlayerCommand::Pause => {
                player.status = PlaybackStatus::Paused;
            }
            PlayerCommand::Stop => {
                player.status = PlaybackStatus::Stopped;
                if let Some(ref mut module) = player.module {
                    module.set_position_seconds(0.0);
                }
            }
            PlayerCommand::Seek(seconds) => {
                if let Some(ref mut module) = player.module {
                    module.set_position_seconds(seconds);
                }
            }
            PlayerCommand::SetInterpolation(filter) => {
                player.interpolation_filter = filter;
                if let Some(ref mut module) = player.module {
                    module.set_interpolation_filter(filter);
                }
            }
            PlayerCommand::SetStereoSeparation(percent) => {
                player.stereo_separation = percent;
                if let Some(ref mut module) = player.module {
                    module.set_stereo_separation(percent);
                }
            }
            PlayerCommand::SetAgcEnabled(enabled) => {
                player.agc_enabled = enabled;
                if let Some(ref mut module) = player.module {
                    module.set_agc_enabled(enabled);
                }
            }
            PlayerCommand::SetHrtfMix(percent) => {
                player.hrtf_mix = percent.clamp(0, 100);
            }
            PlayerCommand::SetVolume(vol) => {
                player.fade_target = vol.clamp(0.0, 1.0);
            }
            PlayerCommand::PlaySample { data, rate_ratio } => {
                if let Some(voice) = player.keyjazz_voices.iter_mut().find(|v| !v.active) {
                    voice.data = data;
                    voice.position = 0.0;
                    voice.rate_ratio = rate_ratio;
                    voice.active = true;
                }
            }
        }
    }
}

fn refresh_visual_snapshot(inner: &Mutex<PlayerInner>, state: &Mutex<PlayerState>) {
    let (status, load_generation, channel_vu, active_samples) = {
        let player = inner.lock().unwrap();
        if player.status == PlaybackStatus::Playing {
            if let Some(module) = player.module.as_ref() {
                (
                    player.status.clone(),
                    player.load_generation,
                    module.channel_vu(),
                    module.active_samples(),
                )
            } else {
                (
                    player.status.clone(),
                    player.load_generation,
                    Vec::new(),
                    Vec::new(),
                )
            }
        } else {
            (
                player.status.clone(),
                player.load_generation,
                Vec::new(),
                Vec::new(),
            )
        }
    };

    let mut snapshot = state.lock().unwrap();
    snapshot.status = status;
    snapshot.load_generation = load_generation;
    snapshot.channel_vu = channel_vu;
    snapshot.active_samples = active_samples;
}

fn mix_keyjazz_voices(data: &mut [f64], voices: &mut [KeyjazzVoice]) {
    let frames = data.len() / 2;
    for voice in voices {
        if !voice.active {
            continue;
        }
        let len = voice.data.len() as f64;
        for i in 0..frames {
            let pos = voice.position;
            if pos >= len - 1.0 {
                voice.active = false;
                break;
            }
            let idx = pos as usize;
            let frac = pos - idx as f64;
            let s0 = voice.data[idx];
            let s1 = voice.data[(idx + 1).min(voice.data.len() - 1)];
            let sample = s0 + (s1 - s0) * frac;
            data[i * 2] += sample * 0.5;
            data[i * 2 + 1] += sample * 0.5;
            voice.position += voice.rate_ratio;
        }
    }
}

/// Mix `dry` (delayed by `latency` samples through `delay_line`) into the wet
/// signal already in `data`. The wet path acquires `latency` samples of
/// startup zero-fill when callback sizes aren't partition multiples; mixing
/// undelayed dry against it would comb-filter, so the dry path is delayed to
/// match. With `latency == 0` and an empty line this is a direct mix.
fn mix_delayed_dry(
    data: &mut [f64],
    dry: &[f64],
    delay_line: &mut Vec<f64>,
    latency: usize,
    wet_gain: f64,
    dry_gain: f64,
) {
    if latency == 0 && delay_line.is_empty() {
        for (wet, d) in data.iter_mut().zip(dry.iter()) {
            *wet = *d * dry_gain + *wet * wet_gain;
        }
        return;
    }
    delay_line.extend_from_slice(dry);
    let take = delay_line.len().saturating_sub(latency).min(data.len());
    let start = data.len() - take;
    for i in 0..take {
        data[start + i] = delay_line[i] * dry_gain + data[start + i] * wet_gain;
    }
    // Positions before `start` hold wet output whose delayed-dry
    // counterparts are still queued in the filling delay line (the wet
    // path's startup zero-fill lands at the tail of `data`, not here):
    // scale by wet gain only since delayed dry history doesn't exist yet.
    for s in &mut data[..start] {
        *s *= wet_gain;
    }
    let remaining = delay_line.len() - take;
    delay_line.copy_within(take.., 0);
    delay_line.truncate(remaining);
}

fn render_audio_f64(
    data: &mut [f64],
    inner: &Mutex<PlayerInner>,
    state: &Mutex<PlayerState>,
    command_rx: &Receiver<PlayerCommand>,
    waveform_buf: &Mutex<Vec<f64>>,
    rate: i32,
) {
    process_audio_commands(inner, command_rx);

    let mut player = inner.lock().unwrap();
    let mut vol = player.volume;
    let fade_target = player.fade_target;
    let mut rendered_audio = false;
    let mut rendered_frames = 0usize;

    if player.status == PlaybackStatus::Playing
        && let Some(ref mut module) = player.module
    {
        let rendered = module.read_interleaved_double_stereo(rate, data);
        rendered_frames = rendered;
        if rendered * 2 < data.len() {
            for sample in &mut data[rendered * 2..] {
                *sample = 0.0;
            }
        }
        if vol != fade_target || vol < 1.0 {
            // Advance the ramp once per FRAME (the step is in per-frame
            // units); stepping per interleaved sample ran the fade twice as
            // fast and gave L/R gains one step apart within each frame.
            let step = 1.0 / (rate as f64 * 0.15);
            for frame in data[..rendered * 2].chunks_exact_mut(2) {
                if (vol - fade_target).abs() > step {
                    vol += if fade_target < vol { -step } else { step };
                } else {
                    vol = fade_target;
                }
                frame[0] *= vol;
                frame[1] *= vol;
            }
        }

        let mut s = state.lock().unwrap();
        s.status = PlaybackStatus::Playing;
        s.position_seconds = module.position_seconds();
        s.duration_seconds = module.duration_seconds();
        s.current_order = module.current_order();
        s.current_row = module.current_row();
        s.current_pattern = module.current_pattern();
        s.bpm = module.current_bpm();
        s.speed = module.current_speed();
        s.load_generation = player.load_generation;
        rendered_audio = true;
    }

    if !rendered_audio {
        for sample in data.iter_mut() {
            *sample = 0.0;
        }
        let mut s = state.lock().unwrap();
        s.status = player.status.clone();
        s.load_generation = player.load_generation;
        if let Some(ref module) = player.module {
            s.position_seconds = module.position_seconds();
            s.duration_seconds = module.duration_seconds();
            s.current_order = module.current_order();
            s.current_row = module.current_row();
            s.current_pattern = module.current_pattern();
            s.bpm = module.current_bpm();
            s.speed = module.current_speed();
        } else {
            s.position_seconds = 0.0;
            s.duration_seconds = 0.0;
            s.current_order = 0;
            s.current_row = 0;
            s.current_pattern = 0;
            s.bpm = 0.0;
            s.speed = 0;
        }
    }

    mix_keyjazz_voices(data, &mut player.keyjazz_voices);

    // HRTF binaural spatialization (headphones mode). The processor is built
    // OFF the audio thread (`spawn_hrtf_init` — parsing the multi-MB SOFA
    // file here used to blow the very first callback's deadline and trigger
    // a spurious buffer-growth/device-bounce at startup); until it lands we
    // simply play dry.
    if player.hrtf_enabled {
        let p = &mut *player; // reborrow for split field access
        let mix = p.hrtf_mix;
        let mut hrtf_failed_now = false;
        if let Some(ref mut processor) = p.hrtf_processor {
            if mix < 100 {
                // Save dry signal before HRTF (no alloc after first callback)
                let len = data.len();
                if p.hrtf_dry_buf.len() < len {
                    p.hrtf_dry_buf.resize(len, 0.0);
                }
                p.hrtf_dry_buf[..len].copy_from_slice(data);
                processor.process(data);
                if processor.is_failed() {
                    hrtf_failed_now = true;
                    data.copy_from_slice(&p.hrtf_dry_buf[..len]);
                } else {
                    let wet_gain = mix as f64 / 100.0;
                    let dry_gain = 1.0 - wet_gain;
                    let latency = processor.latency_samples();
                    mix_delayed_dry(
                        data,
                        &p.hrtf_dry_buf[..len],
                        &mut p.hrtf_dry_delay,
                        latency,
                        wet_gain,
                        dry_gain,
                    );
                }
            } else {
                processor.process(data);
                hrtf_failed_now = processor.is_failed();
            }
        }
        if hrtf_failed_now {
            p.hrtf_enabled = false;
            p.hrtf_processor = None;
            // Drop the dry backlog with the processor; a future replacement
            // starts its latency from zero. `clear` keeps the allocation, so
            // this is safe on the audio thread.
            p.hrtf_dry_delay.clear();
            let mut s = state.lock().unwrap();
            s.hrtf_failed = true;
            s.error = Some("HRTF disabled: renderer error during playback".into());
        }
    }

    player.volume = vol;
    {
        let waveform_status = if rendered_audio {
            PlaybackStatus::Playing
        } else {
            player.status.clone()
        };
        let mut waveform = waveform_buf.lock().unwrap();
        sync_waveform_buffer(&mut waveform, &waveform_status, data, rendered_frames);
    }
}

fn copy_f64_to_f32_output(dst: &mut [f32], src: &[f64]) {
    for (dst, src) in dst.iter_mut().zip(src.iter()) {
        *dst = (*src).clamp(-1.0, 1.0) as f32;
    }
}

struct SdlAudioCallback {
    inner: Arc<Mutex<PlayerInner>>,
    state: Arc<Mutex<PlayerState>>,
    command_rx: Arc<Receiver<PlayerCommand>>,
    waveform: Arc<Mutex<Vec<f64>>>,
    rate: i32,
    render_buffer: Vec<f64>,
    scratch_f32: Vec<f32>,
}

impl AudioCallback<f32> for SdlAudioCallback {
    fn callback(&mut self, stream: &mut AudioStream, requested: i32) {
        // `requested` is the total interleaved f32 sample count SDL3 wants. Feed
        // it in fixed CHUNK-sized slices so each render / HRTF pass stays within
        // the HRTF input ceiling no matter how large a block SDL3 asks for.
        const CHUNK: usize = CALLBACK_CHUNK_FRAMES * 2;
        if self.render_buffer.len() != CHUNK {
            self.render_buffer.resize(CHUNK, 0.0);
        }
        if self.scratch_f32.len() != CHUNK {
            self.scratch_f32.resize(CHUNK, 0.0);
        }
        let mut remaining = requested.max(0) as usize;
        // Keep every slice frame-aligned (even = whole stereo frames). SDL3
        // requests are frame-aligned in practice, so `remaining` stays even and
        // this masking is a no-op; it only guards a pathological odd request,
        // which would otherwise split a frame and swap L/R for the rest of
        // playback. A trailing odd sample (can't happen normally) is left
        // undelivered rather than desyncing the stream.
        while remaining >= 2 {
            let n = remaining.min(CHUNK) & !1;
            let render = &mut self.render_buffer[..n];
            render_audio_f64(
                render,
                &self.inner,
                &self.state,
                &self.command_rx,
                &self.waveform,
                self.rate,
            );
            copy_f64_to_f32_output(&mut self.scratch_f32[..n], render);
            let _ = stream.put_data_f32(&self.scratch_f32[..n]);
            remaining -= n;
        }
    }
}

struct MainThreadSdl {
    _context: sdl3::Sdl,
    // Held to keep the audio subsystem initialized for the Player's lifetime;
    // not read after the stream is opened in `Player::new`.
    _audio: sdl3::AudioSubsystem,
}

// SAFETY: `sdl3::Sdl` and `AudioSubsystem` are `!Send`/`!Sync` because SDL
// internally requires most calls happen on the thread that called `SDL_Init`.
// Player is constructed on the main thread (in `Player::new()`), and nothing
// touches `self.sdl` after construction — the playback stream is opened once
// and lives for the Player's lifetime, so the subsystem is never used across
// threads.
unsafe impl Send for MainThreadSdl {}
unsafe impl Sync for MainThreadSdl {}

/// Owns the live SDL3 playback stream and its boxed callback.
/// `AudioStreamWithCallback` holds a raw `*mut c_void` to the callback box, so
/// it is `!Send`/`!Sync`; this newtype asserts the same invariant as
/// `MainThreadSdl`. The stream is created only in `Player::new()` (main thread)
/// and dropped when the Player is dropped; the audio callback runs on SDL's own
/// audio thread and never touches this handle directly.
struct AudioDeviceHandle(#[allow(dead_code)] AudioStreamWithCallback<SdlAudioCallback>);
unsafe impl Send for AudioDeviceHandle {}
unsafe impl Sync for AudioDeviceHandle {}

pub struct Player {
    inner: Arc<Mutex<PlayerInner>>,
    command_tx: Sender<PlayerCommand>,
    command_rx: Arc<Receiver<PlayerCommand>>,
    state: Arc<Mutex<PlayerState>>,
    waveform: Arc<Mutex<Vec<f64>>>,
    sdl: Option<MainThreadSdl>,
    // Owns the live SDL3 playback stream (and its audio callback) for the
    // Player's lifetime. Never read after construction — held so playback keeps
    // running until the Player is dropped. `None` for the dummy (no-audio)
    // player. Kept behind `Mutex<Option<…>>` to preserve that shape; SDL3 no
    // longer reopens the device, so it is never mutated post-construction.
    #[allow(dead_code)]
    audio_device: Mutex<Option<AudioDeviceHandle>>,
    current_playback_rate: AtomicU32,
}

#[derive(Clone)]
pub struct RenderHandle {
    inner: Arc<Mutex<PlayerInner>>,
}

impl RenderHandle {
    /// Render the live module to a samples buffer.  Holds the player mutex only
    /// for the duration of the render (~seconds), then releases it.  The caller
    /// can encode the returned samples without any mutex contention.
    pub fn render_live_to_samples(
        &self,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
        sample_rate: u32,
        progress: Option<(&crossbeam_channel::Sender<f32>, f32, f32)>,
    ) -> Result<Vec<f64>, String> {
        let mut player = self.inner.lock().unwrap();
        let module = player.module.as_mut().ok_or("No module loaded")?;
        crate::render::render_live_module_to_samples_with_agc(
            module,
            stereo_separation,
            interpolation_filter,
            agc_enabled,
            sample_rate,
            progress,
        )
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_live_to_flac(
        &self,
        output_path: &Path,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
        sample_rate: u32,
        hrtf_mix: i32,
        metadata: &crate::render::AudioMetadata,
    ) -> Result<(), String> {
        let mut player = self.inner.lock().unwrap();
        let module = player.module.as_mut().ok_or("No module loaded")?;
        crate::render::render_live_module_to_flac(
            module,
            output_path,
            stereo_separation,
            interpolation_filter,
            agc_enabled,
            sample_rate,
            hrtf_mix,
            metadata,
        )
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn render_live_to_aac(
        &self,
        output_path: &Path,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
        sample_rate: u32,
        hrtf_mix: i32,
        metadata: &crate::render::AudioMetadata,
    ) -> Result<(), String> {
        let mut player = self.inner.lock().unwrap();
        let module = player.module.as_mut().ok_or("No module loaded")?;
        crate::render::render_live_module_to_aac(
            module,
            output_path,
            stereo_separation,
            interpolation_filter,
            agc_enabled,
            sample_rate,
            hrtf_mix,
            metadata,
        )
    }
}

/// Build the HRTF processor on a background thread and install it into the
/// player when ready. Parsing the embedded SOFA file takes far longer than an
/// audio-callback deadline, so this must never run on the audio thread; the
/// callback plays dry until the processor lands. On failure HRTF is disabled
/// and the failure is surfaced through `PlayerState` so the GUI can uncheck
/// its toggle.
fn spawn_hrtf_init(
    inner: Arc<Mutex<PlayerInner>>,
    state: Arc<Mutex<PlayerState>>,
    rate: u32,
) {
    {
        let mut player = inner.lock().unwrap();
        let already_ready = player.hrtf_processor.is_some() && player.hrtf_rate == rate;
        if !player.hrtf_enabled || already_ready || player.hrtf_init_in_flight || rate == 0 {
            return;
        }
        player.hrtf_init_in_flight = true;
    }
    std::thread::spawn(move || {
        let result = crate::hrtf::HrtfProcessor::try_new(rate);
        let mut player = inner.lock().unwrap();
        player.hrtf_init_in_flight = false;
        match result {
            Ok(processor) => {
                player.hrtf_processor = Some(processor);
                player.hrtf_rate = rate;
                // The replacement's startup latency restarts at zero, so any
                // dry backlog delayed for the old processor would leave the
                // wet/dry mix permanently mis-aligned (comb filtering).
                player.hrtf_dry_delay.clear();
            }
            Err(e) => {
                eprintln!("HRTF init failed: {e}");
                player.hrtf_enabled = false;
                player.hrtf_processor = None;
                player.hrtf_dry_delay.clear();
                let mut s = state.lock().unwrap();
                s.hrtf_failed = true;
                s.error = Some(format!("HRTF disabled: {e}"));
            }
        }
    });
}

fn candidate_playback_rates(preferred: Option<u32>) -> Vec<u32> {
    match preferred {
        Some(rate) => std::iter::once(rate)
            .chain(
                PREFERRED_PLAYBACK_RATES
                    .iter()
                    .copied()
                    .filter(|&r| r < rate),
            )
            .collect(),
        None => PREFERRED_PLAYBACK_RATES.to_vec(),
    }
}

impl Player {
    pub fn new(preferred_rate: Option<u32>) -> Result<Self, String> {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let command_rx = Arc::new(command_rx);
        let inner = Arc::new(Mutex::new(PlayerInner::new()));
        let state = Arc::new(Mutex::new(PlayerState::default()));
        let waveform = Arc::new(Mutex::new(Vec::new()));

        let context = sdl3::init().map_err(|e| format!("SDL3 init failed: {e}"))?;
        let audio = context
            .audio()
            .map_err(|e| format!("SDL3 audio init failed: {e}"))?;

        // Prefer 96 kHz for startup, but keep trying other high and common rates
        // so playback still comes up if a device rejects one. When the user
        // passes --playback-rate, try that first and only fall back to strictly
        // lower rates. Note: SDL3 resamples the source spec to the hardware rate
        // internally, so opening rarely fails on an "unsupported" rate now — the
        // fallback list mainly honors --playback-rate and guards genuine open
        // errors (no device / device busy).
        let candidates = candidate_playback_rates(preferred_rate);
        let mut last_err = String::from("No candidate sample rates available");
        let mut initial_rate = DEFAULT_PLAYBACK_RATE;
        let mut device = None;
        for &rate in &candidates {
            let spec = AudioSpec {
                freq: Some(rate as i32),
                channels: Some(2),
                format: Some(AudioFormat::f32_sys()),
            };
            let callback = SdlAudioCallback {
                inner: inner.clone(),
                state: state.clone(),
                command_rx: command_rx.clone(),
                waveform: waveform.clone(),
                rate: rate as i32,
                render_buffer: Vec::new(),
                scratch_f32: Vec::new(),
            };
            match audio.open_playback_stream::<SdlAudioCallback, f32>(&spec, callback) {
                Ok(dev) => {
                    if let Err(e) = dev.resume() {
                        last_err = e.to_string();
                        continue;
                    }
                    // SDL3 resamples to the device's native rate, so the source
                    // rate we requested IS the rate the renderer runs at — which
                    // is exactly what every downstream consumer (render-rate
                    // defaults, HRTF rate, oscilloscope scaling) wants.
                    initial_rate = rate;
                    device = Some(dev);
                    break;
                }
                Err(e) => {
                    last_err = e.to_string();
                }
            }
        }
        let device = device.ok_or(last_err)?;
        spawn_hrtf_init(inner.clone(), state.clone(), initial_rate);

        Ok(Player {
            inner,
            command_tx,
            command_rx,
            state,
            waveform,
            sdl: Some(MainThreadSdl {
                _context: context,
                _audio: audio,
            }),
            audio_device: Mutex::new(Some(AudioDeviceHandle(device))),
            current_playback_rate: AtomicU32::new(initial_rate),
        })
    }

    /// Create a player without audio output (for when no audio device is available).
    pub fn dummy() -> Self {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let inner = Arc::new(Mutex::new(PlayerInner::new()));
        let state = Arc::new(Mutex::new(PlayerState::default()));

        Player {
            inner,
            command_tx,
            command_rx: Arc::new(command_rx),
            state,
            waveform: Arc::new(Mutex::new(Vec::new())),
            sdl: None,
            audio_device: Mutex::new(None),
            current_playback_rate: AtomicU32::new(0),
        }
    }

    pub fn send(&self, cmd: PlayerCommand) {
        // Only the audio callback drains the queue. The dummy player has no
        // callback, so queuing there would leak every command for the life
        // of the session (PlaySample clones whole sample buffers).
        if self.sdl.is_none() {
            return;
        }
        let _ = self.command_tx.send(cmd);
    }

    pub fn set_volume(&self, vol: f64) {
        self.send(PlayerCommand::SetVolume(vol));
    }

    pub fn state(&self) -> PlayerState {
        self.state.lock().unwrap().clone()
    }

    /// Get the latest rendered audio buffer for oscilloscope display.
    pub fn waveform(&self) -> Vec<f64> {
        self.waveform.lock().unwrap().clone()
    }

    pub fn render_handle(&self) -> RenderHandle {
        RenderHandle {
            inner: self.inner.clone(),
        }
    }

    /// Get direct access to the inner module for remastering.
    /// The caller must hold the lock for the entire remaster operation.
    pub fn with_module<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut Module) -> R,
    {
        let mut player = self.inner.lock().unwrap();
        player.module.as_mut().map(f)
    }

    /// Snapshot of the load-time normalization stash: the pristine native
    /// (< 48 kHz) samples paired with their 48 kHz masters. The GUI remaster
    /// sources its originals from here (AI sees native; live module is 48 kHz).
    /// Empty when normalization is disabled or the module has no sub-48 kHz
    /// samples — callers fall back to reading native samples in that case.
    pub fn sample_stash(&self) -> Vec<crate::remaster::NormalizedSample> {
        self.inner.lock().unwrap().sample_stash.clone()
    }

    pub fn install_prepared_load_with_settings(
        &self,
        prepared: PreparedModuleLoad,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
    ) {
        // Commands queued against the OLD module (seeks, keyjazz notes,
        // transport) must not be applied to the new one by the next callback.
        // Settings-style commands survive: they are intentional regardless of
        // which module is loaded, so they're re-queued after the drain.
        let mut keep: Vec<PlayerCommand> = Vec::new();
        while let Ok(cmd) = self.command_rx.try_recv() {
            match cmd {
                PlayerCommand::SetInterpolation(_)
                | PlayerCommand::SetStereoSeparation(_)
                | PlayerCommand::SetAgcEnabled(_)
                | PlayerCommand::SetVolume(_)
                | PlayerCommand::SetHrtfMix(_) => keep.push(cmd),
                PlayerCommand::Play
                | PlayerCommand::Pause
                | PlayerCommand::Stop
                | PlayerCommand::Seek(_)
                | PlayerCommand::PlaySample { .. } => {}
            }
        }
        for cmd in keep {
            let _ = self.command_tx.send(cmd);
        }
        let mut player = self.inner.lock().unwrap();
        player.stereo_separation = stereo_separation;
        player.interpolation_filter = interpolation_filter;
        player.agc_enabled = agc_enabled;

        let mut state = self.state.lock().unwrap();
        install_prepared_load(&mut player, &mut state, prepared);
    }

    pub fn refresh_visual_state(&self) {
        refresh_visual_snapshot(&self.inner, &self.state);
    }

    /// Retained as a no-op after the SDL3 migration. SDL3's `AudioStream`
    /// buffers internally and decouples the callback from the hardware buffer,
    /// so the callback-gap underrun detection (and the buffer this used to
    /// reset) no longer exists. Kept so the post-render GUI call sites need no
    /// change.
    pub fn clear_stall_underrun(&self) {}

    /// Enable or disable HRTF synchronously. This bypasses the command queue
    /// on purpose: the queue is only drained by the audio callback, so a
    /// queued enable would still read as disabled when `spawn_hrtf_init`
    /// checks it and the init would silently no-op. An explicit enable also
    /// clears the `hrtf_failed` latch so the user can retry after a
    /// transient init/render error; a failure during that retry latches
    /// again until the next explicit enable.
    pub fn set_hrtf_enabled(&self, enabled: bool) {
        self.inner.lock().unwrap().hrtf_enabled = enabled;
        if enabled {
            self.state.lock().unwrap().hrtf_failed = false;
            self.ensure_hrtf_ready();
        }
    }

    /// Kick (or re-kick) background HRTF initialization at the current
    /// playback rate. Called when the user enables HRTF; the audio callback
    /// itself never builds the processor.
    pub fn ensure_hrtf_ready(&self) {
        let rate = self.current_playback_rate.load(Ordering::Relaxed);
        spawn_hrtf_init(self.inner.clone(), self.state.clone(), rate);
    }

    pub fn current_playback_rate(&self) -> u32 {
        self.current_playback_rate
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn output_format_label(&self) -> &'static str {
        "F32"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn seeded_keyjazz_inner() -> Mutex<PlayerInner> {
        let mut inner = PlayerInner::new();
        inner.status = PlaybackStatus::Playing;
        inner.keyjazz_voices[0].data = vec![0.2, -0.4, 0.6, -0.8];
        inner.keyjazz_voices[0].position = 0.0;
        inner.keyjazz_voices[0].rate_ratio = 1.0;
        inner.keyjazz_voices[0].active = true;
        Mutex::new(inner)
    }

    #[test]
    fn rendered_audio_populates_waveform_buffer_from_rendered_frames_only() {
        let mut waveform = Vec::new();

        sync_waveform_buffer(
            &mut waveform,
            &PlaybackStatus::Playing,
            &[0.1, -0.1, 0.2, -0.2, 9.0, 9.0],
            2,
        );

        assert_eq!(waveform, vec![0.1, -0.1, 0.2, -0.2]);
    }

    #[test]
    fn pause_preserves_last_waveform() {
        let mut waveform = vec![0.2, -0.2, 0.4, -0.4];

        sync_waveform_buffer(&mut waveform, &PlaybackStatus::Paused, &[], 0);

        assert_eq!(waveform, vec![0.2, -0.2, 0.4, -0.4]);
    }

    #[test]
    fn stop_clears_waveform() {
        let mut waveform = vec![0.2, -0.2, 0.4, -0.4];

        sync_waveform_buffer(&mut waveform, &PlaybackStatus::Stopped, &[], 0);

        assert!(waveform.is_empty());
    }

    #[test]
    fn waveform_sync_reuses_existing_allocation() {
        let mut waveform = Vec::with_capacity(16);
        waveform.extend_from_slice(&[9.0, 9.0, 9.0, 9.0]);
        let initial_ptr = waveform.as_ptr();
        let initial_capacity = waveform.capacity();

        sync_waveform_buffer(
            &mut waveform,
            &PlaybackStatus::Playing,
            &[0.1, -0.1, 0.2, -0.2],
            2,
        );

        assert_eq!(waveform, vec![0.1, -0.1, 0.2, -0.2]);
        assert_eq!(waveform.capacity(), initial_capacity);
        assert!(ptr::eq(waveform.as_ptr(), initial_ptr));
    }

    #[test]
    fn installing_prepared_load_updates_generation_and_clears_error() {
        let mut inner = PlayerInner::new();
        let mut state = PlayerState {
            error: Some("old error".into()),
            ..PlayerState::default()
        };
        let prepared = prepare_module_load_from_bytes(std::fs::read("mods/module76.s3m").unwrap())
            .expect("module should prepare");

        install_prepared_load(&mut inner, &mut state, prepared);

        assert!(inner.module.is_some());
        assert_eq!(inner.status, PlaybackStatus::Playing);
        assert_eq!(inner.load_generation, 1);
        assert_eq!(state.error, None);
        assert_eq!(state.load_generation, 1);
    }

    #[test]
    fn audio_callback_leaves_visual_snapshot_updates_to_non_rt_path() {
        let inner = seeded_keyjazz_inner();
        let state = Mutex::new(PlayerState {
            channel_vu: vec![(0.5, 0.5)],
            active_samples: vec![7],
            ..PlayerState::default()
        });
        let waveform = Mutex::new(Vec::new());
        let (_tx, rx) = crossbeam_channel::unbounded();
        let mut f64_output = vec![0.0f64; 8];

        render_audio_f64(&mut f64_output, &inner, &state, &rx, &waveform, 48_000);

        let snapshot = state.lock().unwrap().clone();
        assert_eq!(snapshot.channel_vu, vec![(0.5, 0.5)]);
        assert_eq!(snapshot.active_samples, vec![7]);
    }

    #[test]
    fn non_rt_visual_refresh_clears_snapshot_when_not_playing() {
        let inner = Mutex::new(PlayerInner::new());
        let state = Mutex::new(PlayerState {
            channel_vu: vec![(0.5, 0.5)],
            active_samples: vec![7],
            ..PlayerState::default()
        });

        refresh_visual_snapshot(&inner, &state);

        let snapshot = state.lock().unwrap().clone();
        assert!(snapshot.channel_vu.is_empty());
        assert!(snapshot.active_samples.is_empty());
    }

    #[test]
    fn f32_output_matches_f64_core() {
        let inner_f64 = seeded_keyjazz_inner();
        let inner_f32 = seeded_keyjazz_inner();
        let state_f64 = Mutex::new(PlayerState::default());
        let state_f32 = Mutex::new(PlayerState::default());
        let waveform_f64 = Mutex::new(Vec::new());
        let waveform_f32 = Mutex::new(Vec::new());
        let (_tx, rx) = crossbeam_channel::unbounded();

        let mut f64_output = vec![0.0f64; 8];
        render_audio_f64(
            &mut f64_output,
            &inner_f64,
            &state_f64,
            &rx,
            &waveform_f64,
            48_000,
        );

        let mut f64_for_f32 = vec![0.0f64; 8];
        render_audio_f64(
            &mut f64_for_f32,
            &inner_f32,
            &state_f32,
            &rx,
            &waveform_f32,
            48_000,
        );
        let mut f32_output = vec![0.0f32; 8];
        copy_f64_to_f32_output(&mut f32_output, &f64_for_f32);

        let expected: Vec<f32> = f64_output.iter().map(|&sample| sample as f32).collect();
        assert_eq!(f32_output, expected);
    }

    #[test]
    fn f32_output_clamps_to_valid_range() {
        let overshooting = vec![1.5_f64, -1.5, 0.5, -0.5];
        let mut output = vec![0.0f32; 4];
        copy_f64_to_f32_output(&mut output, &overshooting);
        assert_eq!(output, vec![1.0f32, -1.0, 0.5, -0.5]);
    }

    #[test]
    fn playback_candidate_rates_prefer_96khz_first() {
        assert_eq!(PREFERRED_PLAYBACK_RATES, &[96_000, 88_200, 48_000, 44_100]);
    }

    #[test]
    fn callback_chunk_frames_is_1024() {
        assert_eq!(CALLBACK_CHUNK_FRAMES, 1024);
    }

    #[test]
    fn candidate_rates_with_preferred_excludes_higher_rates() {
        assert_eq!(candidate_playback_rates(Some(48_000)), vec![48_000, 44_100]);
        assert_eq!(
            candidate_playback_rates(Some(88_200)),
            vec![88_200, 48_000, 44_100]
        );
        assert_eq!(candidate_playback_rates(Some(44_100)), vec![44_100]);
    }

    #[test]
    fn candidate_rates_with_unusual_preferred_still_orders_lower_fallbacks() {
        assert_eq!(
            candidate_playback_rates(Some(48_001)),
            vec![48_001, 48_000, 44_100]
        );
    }

    #[test]
    fn candidate_rates_none_matches_default_preferred_list() {
        assert_eq!(
            candidate_playback_rates(None),
            PREFERRED_PLAYBACK_RATES.to_vec()
        );
    }
}
