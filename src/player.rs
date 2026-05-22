// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use crossbeam_channel::{Receiver, Sender};
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::openmpt::{
    AgcProfile, DEFAULT_AGC_ENABLED, DEFAULT_AGC_PROFILE, DEFAULT_INTERPOLATION_FILTER_LENGTH,
    DEFAULT_STEREO_SEPARATION_PERCENT, Module,
};

const DEFAULT_PLAYBACK_RATE: u32 = 48_000;
const PREFERRED_PLAYBACK_RATES: &[u32] = &[96_000, 88_200, 48_000, 44_100];
const DEFAULT_BUFFER_FRAMES: u32 = 1024;
/// Maximum buffer size for auto-growth (16384 frames ≈ 341ms at 48kHz).
const MAX_BUFFER_FRAMES: u32 = 16384;
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
    SetHrtfEnabled(bool),
    SetHrtfMix(i32),
    PlaySample { data: Vec<f64>, rate_ratio: f64 },
}

pub struct PreparedModuleLoad {
    file_data: Vec<u8>,
    module: Module,
    file_size_bytes: u64,
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

    let module = Module::from_memory(&file_data).map_err(|e| format!("Load failed: {e}"))?;
    Ok(PreparedModuleLoad {
        file_data,
        module,
        file_size_bytes,
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
    hrtf_dry_buf: Vec<f64>,
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
            hrtf_dry_buf: Vec::new(),
        }
    }
}

fn install_prepared_load(
    player: &mut PlayerInner,
    state: &mut PlayerState,
    prepared: PreparedModuleLoad,
) {
    let PreparedModuleLoad {
        file_data,
        mut module,
        file_size_bytes: _,
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
            PlayerCommand::SetHrtfEnabled(enabled) => {
                player.hrtf_enabled = enabled;
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
            let step = 1.0 / (rate as f64 * 0.15);
            for sample in &mut data[..rendered * 2] {
                if (vol - fade_target).abs() > step {
                    vol += if fade_target < vol { -step } else { step };
                } else {
                    vol = fade_target;
                }
                *sample *= vol;
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

    // HRTF binaural spatialization (headphones mode)
    if player.hrtf_enabled {
        if player.hrtf_processor.is_none() {
            match crate::hrtf::HrtfProcessor::try_new(rate as u32) {
                Ok(p) => player.hrtf_processor = Some(p),
                Err(e) => {
                    eprintln!("HRTF init failed: {e}");
                    player.hrtf_enabled = false;
                }
            }
        }
        let p = &mut *player; // reborrow for split field access
        let mix = p.hrtf_mix;
        if let Some(ref mut processor) = p.hrtf_processor {
            if mix < 100 {
                // Save dry signal before HRTF (no alloc after first callback)
                let len = data.len();
                if p.hrtf_dry_buf.len() < len {
                    p.hrtf_dry_buf.resize(len, 0.0);
                }
                p.hrtf_dry_buf[..len].copy_from_slice(data);
                processor.process(data);
                let wet_gain = mix as f64 / 100.0;
                let dry_gain = 1.0 - wet_gain;
                for (wet, dry) in data.iter_mut().zip(&p.hrtf_dry_buf[..len]) {
                    *wet = *dry * dry_gain + *wet * wet_gain;
                }
            } else {
                processor.process(data);
            }
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
    underrun_flag: Arc<AtomicBool>,
    last_callback_nanos: Arc<AtomicU64>,
    epoch: Instant,
    rate: i32,
    render_buffer: Vec<f64>,
}

impl AudioCallback for SdlAudioCallback {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        if self.render_buffer.len() != out.len() {
            self.render_buffer.resize(out.len(), 0.0);
        }
        check_underrun(
            out.len(),
            self.rate,
            &self.epoch,
            &self.last_callback_nanos,
            &self.underrun_flag,
        );
        render_audio_f64(
            &mut self.render_buffer,
            &self.inner,
            &self.state,
            &self.command_rx,
            &self.waveform,
            self.rate,
        );
        copy_f64_to_f32_output(out, &self.render_buffer);
    }
}

struct MainThreadSdl {
    _context: sdl2::Sdl,
    audio: sdl2::AudioSubsystem,
}

// SAFETY: `sdl2::Sdl` and `AudioSubsystem` are `!Send`/`!Sync` because SDL
// internally requires most calls happen on the thread that called `SDL_Init`.
// Player is constructed on the main thread (in `Player::new()`), and the only
// method that touches `self.sdl` after construction is `check_and_grow_buffer()`
// / `try_build_and_play()`, called exclusively from the iced `Message::Tick`
// handler — also the main thread.  The `audio_device` field (which lives
// outside this wrapper) is `Send`+`Sync` on its own and is safe to drop from
// any thread.
unsafe impl Send for MainThreadSdl {}
unsafe impl Sync for MainThreadSdl {}

/// Check for buffer underrun by comparing the gap between consecutive callback
/// invocations against the expected buffer duration.  Sets `underrun_flag` when
/// the gap exceeds 2× the buffer duration (generous to avoid false positives).
fn check_underrun(
    sample_count: usize,
    rate: i32,
    epoch: &Instant,
    last_callback_nanos: &AtomicU64,
    underrun_flag: &AtomicBool,
) {
    let now_nanos = Instant::now().duration_since(*epoch).as_nanos() as u64;
    let prev = last_callback_nanos.swap(now_nanos, Ordering::Relaxed);
    if prev == 0 {
        return; // first call — no previous timestamp to compare
    }
    let elapsed_nanos = now_nanos.saturating_sub(prev);
    let frames = sample_count / 2; // stereo
    let expected_nanos = (frames as u64) * 1_000_000_000 / (rate as u64);
    if elapsed_nanos > expected_nanos * 2 {
        underrun_flag.store(true, Ordering::Relaxed);
    }
}

pub struct Player {
    inner: Arc<Mutex<PlayerInner>>,
    command_tx: Sender<PlayerCommand>,
    command_rx: Arc<Receiver<PlayerCommand>>,
    state: Arc<Mutex<PlayerState>>,
    waveform: Arc<Mutex<Vec<f64>>>,
    sdl: Option<MainThreadSdl>,
    audio_device: Mutex<Option<sdl2::audio::AudioDevice<SdlAudioCallback>>>,
    current_playback_rate: AtomicU32,
    buffer_frames: AtomicU32,
    underrun_flag: Arc<AtomicBool>,
    last_callback_nanos: Arc<AtomicU64>,
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

        let context = sdl2::init().map_err(|e| format!("SDL2 init failed: {e}"))?;
        let audio = context
            .audio()
            .map_err(|e| format!("SDL2 audio init failed: {e}"))?;

        let underrun_flag = Arc::new(AtomicBool::new(false));
        let last_callback_nanos = Arc::new(AtomicU64::new(0));

        // Prefer 96 kHz for startup, but keep trying other high and common rates
        // so playback still comes up on devices that do not support it. When the
        // user passes --playback-rate, try that first and only fall back to
        // strictly lower rates so an unsupported Bluetooth-friendly rate never
        // silently upgrades back to 96 kHz.
        let candidates = candidate_playback_rates(preferred_rate);
        let mut last_err = String::from("No candidate sample rates available");
        let mut initial_rate = DEFAULT_PLAYBACK_RATE;
        let mut device = None;
        for &rate in &candidates {
            let spec = AudioSpecDesired {
                freq: Some(rate as i32),
                channels: Some(2),
                samples: Some(DEFAULT_BUFFER_FRAMES as u16),
            };
            last_callback_nanos.store(0, Ordering::Relaxed);
            let cb_inner = inner.clone();
            let cb_state = state.clone();
            let cb_rx = command_rx.clone();
            let cb_waveform = waveform.clone();
            let cb_uf = underrun_flag.clone();
            let cb_lcn = last_callback_nanos.clone();
            match audio.open_playback(None, &spec, |actual| {
                if actual.channels != 2 {
                    eprintln!(
                        "SDL2: requested 2 channels but got {} — audio may be broken",
                        actual.channels
                    );
                }
                if actual.freq != rate as i32 {
                    eprintln!("SDL2: requested {}Hz but got {}Hz", rate, actual.freq);
                }
                SdlAudioCallback {
                    inner: cb_inner,
                    state: cb_state,
                    command_rx: cb_rx,
                    waveform: cb_waveform,
                    underrun_flag: cb_uf,
                    last_callback_nanos: cb_lcn,
                    epoch: Instant::now(),
                    rate: actual.freq,
                    render_buffer: Vec::new(),
                }
            }) {
                Ok(dev) => {
                    dev.resume();
                    initial_rate = rate;
                    device = Some(dev);
                    break;
                }
                Err(e) => {
                    last_err = e;
                }
            }
        }
        let device = device.ok_or(last_err)?;

        Ok(Player {
            inner,
            command_tx,
            command_rx,
            state,
            waveform,
            sdl: Some(MainThreadSdl {
                _context: context,
                audio,
            }),
            audio_device: Mutex::new(Some(device)),
            current_playback_rate: AtomicU32::new(initial_rate),
            buffer_frames: AtomicU32::new(DEFAULT_BUFFER_FRAMES),
            underrun_flag,
            last_callback_nanos,
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
            buffer_frames: AtomicU32::new(0),
            underrun_flag: Arc::new(AtomicBool::new(false)),
            last_callback_nanos: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn send(&self, cmd: PlayerCommand) {
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

    pub fn install_prepared_load_with_settings(
        &self,
        prepared: PreparedModuleLoad,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
    ) {
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

    /// Check for audio underruns and double the buffer size if one occurred.
    /// Returns `Some(new_frames)` if the buffer was grown, `None` otherwise.
    pub fn check_and_grow_buffer(&self) -> Option<u32> {
        if !self.underrun_flag.swap(false, Ordering::Relaxed) {
            return None;
        }

        self.sdl.as_ref()?;
        let current = self.buffer_frames.load(Ordering::Relaxed);
        let new_frames = next_buffer_frames(current);
        let new_frames = new_frames.min(MAX_BUFFER_FRAMES);
        if new_frames == current {
            return None; // already at max
        }

        let rate = self.current_playback_rate.load(Ordering::Relaxed);
        let mut dev_guard = self.audio_device.lock().unwrap();
        // Drop the old device before opening the replacement.
        *dev_guard = None;

        // Try the larger buffer.  If the driver rejects it, fall back to the
        // previous size so playback is restored rather than lost entirely.
        let (device, frames) = self
            .try_build_and_play(rate, new_frames)
            .or_else(|e| {
                eprintln!(
                    "Buffer resize to {new_frames} failed ({e}), restoring {current}-frame buffer"
                );
                self.try_build_and_play(rate, current)
            })
            .or_else(|e| {
                eprintln!("Restore at {current} also failed ({e}), trying default buffer");
                self.try_build_and_play(rate, 0).map(|(s, _)| (s, 0))
            })
            .ok()?;

        self.buffer_frames.store(frames, Ordering::Relaxed);
        *dev_guard = Some(device);
        if frames != current {
            eprintln!("Audio underrun detected — buffer increased to {frames} frames");
        }
        Some(frames)
    }

    /// Open an SDL2 audio device and resume playback, returning the device and
    /// the buffer size used.
    fn try_build_and_play(
        &self,
        rate: u32,
        buffer_frames: u32,
    ) -> Result<(sdl2::audio::AudioDevice<SdlAudioCallback>, u32), String> {
        let sdl = self.sdl.as_ref().ok_or("No SDL audio subsystem")?;
        let samples = if buffer_frames > 0 {
            Some(u16::try_from(buffer_frames).unwrap_or(u16::MAX))
        } else {
            None
        };
        let spec = AudioSpecDesired {
            freq: Some(rate as i32),
            channels: Some(2),
            samples,
        };
        self.last_callback_nanos.store(0, Ordering::Relaxed);
        let inner = self.inner.clone();
        let state = self.state.clone();
        let command_rx = self.command_rx.clone();
        let waveform = self.waveform.clone();
        let underrun_flag = self.underrun_flag.clone();
        let last_callback_nanos = self.last_callback_nanos.clone();
        let device = sdl
            .audio
            .open_playback(None, &spec, |actual| SdlAudioCallback {
                inner,
                state,
                command_rx,
                waveform,
                underrun_flag,
                last_callback_nanos,
                epoch: Instant::now(),
                rate: actual.freq,
                render_buffer: Vec::new(),
            })?;
        device.resume();
        Ok((device, buffer_frames))
    }

    pub fn current_playback_rate(&self) -> u32 {
        self.current_playback_rate
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn output_format_label(&self) -> &'static str {
        "F32"
    }
}

fn next_buffer_frames(current: u32) -> u32 {
    if current == 0 {
        DEFAULT_BUFFER_FRAMES
    } else {
        current.saturating_mul(2)
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
        let prepared = prepare_module_load_from_bytes(std::fs::read("mods/2ND_PM.S3M").unwrap())
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
    fn default_buffer_frames_is_1024() {
        assert_eq!(DEFAULT_BUFFER_FRAMES, 1024);
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

    #[test]
    fn next_buffer_frames_grows_from_default_baseline() {
        assert_eq!(next_buffer_frames(0), DEFAULT_BUFFER_FRAMES);
        assert_eq!(next_buffer_frames(DEFAULT_BUFFER_FRAMES), 2048);
        assert_eq!(next_buffer_frames(2048), 4096);
    }
}
