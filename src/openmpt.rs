// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use std::ffi::{CStr, CString, c_char, c_double, c_float, c_int, c_void};

// Opaque type for the module handle
#[repr(C)]
pub struct OpenmptModule {
    _opaque: [u8; 0],
}

// Render parameter constants
pub const RENDER_MASTERGAIN_MILLIBEL: c_int = 1;
pub const RENDER_STEREOSEPARATION_PERCENT: c_int = 2;
pub const RENDER_INTERPOLATIONFILTER_LENGTH: c_int = 3;
pub const RENDER_VOLUMERAMPING_STRENGTH: c_int = 4;

pub const DEFAULT_STEREO_SEPARATION_PERCENT: i32 = 66;
pub const MOD_STEREO_SEPARATION_PERCENT: i32 = 66;
pub const DEFAULT_INTERPOLATION_FILTER_LENGTH: i32 = 64;

/// Format-aware stereo-separation default. Currently MOD and non-MOD share 66%,
/// but the hook is kept so Amiga-style hard panning can diverge later without
/// touching every call site.
pub fn effective_stereo_separation(path: &std::path::Path, default: i32) -> i32 {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("mod"))
    {
        MOD_STEREO_SEPARATION_PERCENT
    } else {
        default
    }
}
pub const DEFAULT_VOLUMERAMPING_STRENGTH: i32 = 10;
pub const DEFAULT_AGC_ENABLED: bool = true;
pub const DEFAULT_AGC_PROFILE: AgcProfile = AgcProfile::Gentle;

const AGC_PROFILE_STOCK: c_int = 0;
const AGC_PROFILE_GENTLE: c_int = 1;
const SAMPLE_FORMAT_INT8: c_int = 0;
const SAMPLE_FORMAT_INT16: c_int = 1;
const SAMPLE_FORMAT_FLOAT32: c_int = 2;
const SAMPLE_FORMAT_FLOAT64: c_int = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgcProfile {
    Stock,
    Gentle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Int8,
    Int16,
    Float32,
    Float64,
}

impl SampleFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Int8 => "8",
            Self::Int16 => "16",
            Self::Float32 => "32f",
            Self::Float64 => "64f",
        }
    }
}

// Initial ctl struct for module creation
#[repr(C)]
pub struct OpenmptModuleInitialCtl {
    pub ctl: *const c_char,
    pub value: *const c_char,
}

type LogFunc = Option<unsafe extern "C" fn(*const c_char, *mut c_void)>;
type ErrFunc = Option<unsafe extern "C" fn(c_int, *mut c_void) -> c_int>;

#[link(name = "openmpt", kind = "static")]
#[link(name = "stdc++")]
#[allow(dead_code)]
unsafe extern "C" {
    fn openmpt_free_string(str: *const c_char);

    fn openmpt_module_create_from_memory2(
        filedata: *const c_void,
        filesize: usize,
        logfunc: LogFunc,
        loguser: *mut c_void,
        errfunc: ErrFunc,
        erruser: *mut c_void,
        error: *mut c_int,
        error_message: *mut *const c_char,
        ctls: *const OpenmptModuleInitialCtl,
    ) -> *mut OpenmptModule;

    fn openmpt_module_destroy(module: *mut OpenmptModule);

    // Rendering
    fn openmpt_module_read_interleaved_float_stereo(
        module: *mut OpenmptModule,
        samplerate: i32,
        count: usize,
        interleaved_stereo: *mut c_float,
    ) -> usize;

    fn openmpt_module_read_interleaved_double_stereo(
        module: *mut OpenmptModule,
        samplerate: i32,
        count: usize,
        interleaved_stereo: *mut c_double,
    ) -> usize;

    // Metadata
    fn openmpt_module_get_metadata(module: *mut OpenmptModule, key: *const c_char)
    -> *const c_char;
    #[cfg(test)]
    fn openmpt_quinlight_get_nativefloat_size() -> u32;
    fn openmpt_module_get_duration_seconds(module: *mut OpenmptModule) -> f64;
    #[cfg(test)]
    fn openmpt_module_error_get_last(module: *mut OpenmptModule) -> c_int;
    #[cfg(test)]
    fn openmpt_module_error_get_last_message(module: *mut OpenmptModule) -> *const c_char;
    fn openmpt_module_get_num_channels(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_num_orders(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_num_patterns(module: *mut OpenmptModule) -> i32;
    #[allow(dead_code)]
    fn openmpt_module_get_order_pattern(module: *mut OpenmptModule, order: i32) -> i32;
    fn openmpt_module_get_num_instruments(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_num_samples(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_sample_name(module: *mut OpenmptModule, index: i32) -> *const c_char;
    fn openmpt_module_get_instrument_name(module: *mut OpenmptModule, index: i32) -> *const c_char;

    // Playback control
    fn openmpt_module_get_position_seconds(module: *mut OpenmptModule) -> f64;
    fn openmpt_module_set_position_seconds(module: *mut OpenmptModule, seconds: f64) -> f64;
    fn openmpt_module_get_current_order(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_current_row(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_current_pattern(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_set_repeat_count(module: *mut OpenmptModule, repeat_count: i32) -> c_int;
    fn openmpt_module_set_render_param(
        module: *mut OpenmptModule,
        param: c_int,
        value: i32,
    ) -> c_int;
    fn openmpt_module_get_render_param(
        module: *mut OpenmptModule,
        param: c_int,
        value: *mut i32,
    ) -> c_int;
    #[cfg(test)]
    fn openmpt_module_set_test_preamp(module: *mut OpenmptModule, preamp: i32) -> c_int;

    // VU metering
    fn openmpt_module_get_current_channel_vu_left(
        module: *mut OpenmptModule,
        channel: i32,
    ) -> c_double;
    fn openmpt_module_get_current_channel_vu_right(
        module: *mut OpenmptModule,
        channel: i32,
    ) -> c_double;

    // Quinlight sample data extensions
    fn openmpt_module_get_sample_rate(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_length_frames(module: *mut OpenmptModule, index: i32) -> i64;
    fn openmpt_module_get_sample_channels(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_c5_speed(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_relative_tone(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_fine_tune(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_default_volume(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_has_sample_default_pan(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_default_pan(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_instrument_keyboard_sample(
        module: *mut OpenmptModule,
        instrument_index: i32,
        note: i32,
    ) -> i32;
    fn openmpt_module_read_sample_data(
        module: *mut OpenmptModule,
        index: i32,
        buffer: *mut c_double,
        buffer_frames: i64,
    ) -> i64;
    fn openmpt_module_replace_sample_data(
        module: *mut OpenmptModule,
        index: i32,
        data: *const c_double,
        length_frames: i64,
        channels: i32,
        new_sample_rate: i32,
    ) -> c_int;
    fn openmpt_module_replace_sample_data_raw(
        module: *mut OpenmptModule,
        index: i32,
        data: *const c_double,
        length_frames: i64,
        channels: i32,
        new_sample_rate: i32,
    ) -> c_int;
    fn openmpt_module_set_sample_loop_points(
        module: *mut OpenmptModule,
        index: i32,
        loop_start: i64,
        loop_end: i64,
        loop_mode: i32,
        sustain_start: i64,
        sustain_end: i64,
        sustain_mode: i32,
    ) -> c_int;
    fn openmpt_module_refresh_channels_for_sample(module: *mut OpenmptModule, index: i32) -> c_int;
    fn openmpt_module_get_linear_slides(module: *mut OpenmptModule) -> c_int;
    fn openmpt_module_set_linear_slides(module: *mut OpenmptModule, enabled: c_int) -> c_int;
    fn openmpt_module_get_agc_enabled(module: *mut OpenmptModule) -> c_int;
    fn openmpt_module_set_agc_enabled(module: *mut OpenmptModule, enabled: c_int) -> c_int;
    fn openmpt_module_get_agc_profile(module: *mut OpenmptModule) -> c_int;
    fn openmpt_module_set_agc_profile(module: *mut OpenmptModule, profile: c_int) -> c_int;
    fn openmpt_module_get_sample_bits_per_sample(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_sample_format(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_has_sample_loop(module: *mut OpenmptModule, index: i32) -> c_int;
    fn openmpt_module_get_sample_loop_start(module: *mut OpenmptModule, index: i32) -> i64;
    fn openmpt_module_get_sample_loop_end(module: *mut OpenmptModule, index: i32) -> i64;
    fn openmpt_module_get_sample_loop_mode(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_has_sample_sustain_loop(module: *mut OpenmptModule, index: i32) -> c_int;
    fn openmpt_module_get_sample_sustain_loop_start(module: *mut OpenmptModule, index: i32) -> i64;
    fn openmpt_module_get_sample_sustain_loop_end(module: *mut OpenmptModule, index: i32) -> i64;
    fn openmpt_module_get_sample_sustain_loop_mode(module: *mut OpenmptModule, index: i32) -> i32;
    fn openmpt_module_get_pattern_num_rows(module: *mut OpenmptModule, pattern: i32) -> i32;
    fn openmpt_module_get_pattern_rows_per_beat(module: *mut OpenmptModule, pattern: i32) -> i32;
    fn openmpt_module_get_pattern_row_channel_command(
        module: *mut OpenmptModule,
        pattern: i32,
        row: i32,
        channel: i32,
        command: c_int,
    ) -> u8;
    fn openmpt_module_set_pattern_row_channel_command(
        module: *mut OpenmptModule,
        pattern: i32,
        row: i32,
        channel: i32,
        command: c_int,
        value: u8,
    ) -> c_int;
    fn openmpt_module_get_current_channel_sample(module: *mut OpenmptModule, channel: i32) -> i32;
    #[allow(dead_code)]
    fn openmpt_module_save_to_memory(
        module: *mut OpenmptModule,
        buffer: *mut u8,
        buffer_size: i64,
    ) -> i64;
    fn openmpt_module_save_loaded_format_to_memory(
        module: *mut OpenmptModule,
        buffer: *mut u8,
        buffer_size: i64,
    ) -> i64;
    fn openmpt_module_save_best_format_to_memory(
        module: *mut OpenmptModule,
        buffer: *mut u8,
        buffer_size: i64,
    ) -> i64;
    fn openmpt_module_get_loaded_format_extension(module: *mut OpenmptModule) -> *const c_char;
    fn openmpt_module_get_best_save_format_extension(module: *mut OpenmptModule) -> *const c_char;

    // Pattern display (standard libopenmpt API)
    fn openmpt_module_format_pattern_row_channel(
        module: *mut OpenmptModule,
        pattern: i32,
        row: i32,
        channel: i32,
        width: usize,
        pad: c_int,
    ) -> *const c_char;
    fn openmpt_module_highlight_pattern_row_channel(
        module: *mut OpenmptModule,
        pattern: i32,
        row: i32,
        channel: i32,
        width: usize,
        pad: c_int,
    ) -> *const c_char;
    fn openmpt_module_get_current_estimated_bpm(module: *mut OpenmptModule) -> f64;
    fn openmpt_module_get_current_tempo2(module: *mut OpenmptModule) -> f64;
    fn openmpt_module_get_current_speed(module: *mut OpenmptModule) -> i32;
    fn openmpt_module_get_channel_name(module: *mut OpenmptModule, index: i32) -> *const c_char;
    fn openmpt_module_quinlight_test_get_note_from_period(
        module: *mut OpenmptModule,
        period: f64,
        fine_tune: c_int,
        c5speed: f64,
    ) -> f64;
    fn openmpt_module_quinlight_test_get_period_from_note(
        module: *mut OpenmptModule,
        note: u32,
        fine_tune: c_int,
        c5speed: f64,
    ) -> f64;
    fn openmpt_module_quinlight_get_freq_from_period(
        module: *mut OpenmptModule,
        period: f64,
        c5speed: f64,
    ) -> f64;
    fn openmpt_quinlight_test_apply_linear_pitch_slide(
        target: f64,
        total_amount: c_int,
        periods_are_frequencies: c_int,
    ) -> f64;
    fn openmpt_module_quinlight_test_get_current_channel_period(
        module: *mut OpenmptModule,
        channel: c_int,
    ) -> f64;
    fn openmpt_module_quinlight_test_get_current_channel_frequency(
        module: *mut OpenmptModule,
        channel: c_int,
    ) -> f64;
    fn openmpt_module_quinlight_test_get_current_channel_increment(
        module: *mut OpenmptModule,
        channel: c_int,
    ) -> f64;

    // Ctl get/set for floating-point parameters (beta-shear tuning, etc.)
    fn openmpt_module_ctl_set_floatingpoint(
        module: *mut OpenmptModule,
        ctl: *const c_char,
        value: c_double,
    ) -> c_int;
    fn openmpt_module_ctl_get_floatingpoint(
        module: *mut OpenmptModule,
        ctl: *const c_char,
    ) -> c_double;

    // AVX2 SIMD kernel for stereo 64-tap dot product (Aniso64AVX2.cpp)
    fn aniso64_dot_stereo_avx2(
        kernel: *const c_double,
        samples: *const c_double,
        out_l: *mut c_double,
        out_r: *mut c_double,
    );
}

#[cfg(test)]
unsafe extern "C" {
    fn openmpt_quinlight_test_apply_continuous_linear_pitch_slide(
        target: c_double,
        total_amount: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_apply_it_linear_pitch_slide_reference(
        target: c_double,
        total_amount: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_pitch_envelope_factor(
        envval: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_pitch_envelope_reference_factor(
        envval: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_it_arpeggio_factor(
        semitones: u32,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_it_arpeggio_reference_factor(
        semitones: u32,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_it_autovibrato_factor(
        vdelta: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_it_autovibrato_reference_factor(
        vdelta: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_linear_autovibrato_factor(
        n: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_linear_autovibrato_reference_factor(
        n: c_int,
        periods_are_frequencies: c_int,
    ) -> c_double;
    fn openmpt_quinlight_test_microtuning_factor(finetune: c_int) -> c_double;
    fn openmpt_quinlight_test_hertz_from_note(note: u32, c5speed: c_double) -> c_double;
    fn openmpt_quinlight_test_reference_hertz_from_note(note: u32, c5speed: c_double) -> c_double;
    fn openmpt_quinlight_test_xm_linear_freq_from_period(period: u32) -> c_double;
    fn openmpt_quinlight_test_reference_xm_linear_freq_from_period(period: u32) -> c_double;
}

/// Helper: copy a libopenmpt string to a Rust String and free the original
unsafe fn openmpt_string_to_rust(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
    unsafe { openmpt_free_string(ptr) };
    Some(s)
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn nativefloat_size_bytes() -> usize {
    unsafe { openmpt_quinlight_get_nativefloat_size() as usize }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SampleInfo {
    pub index: i32,
    pub name: String,
    pub rate: i32,
    pub length_frames: i64,
    pub channels: i32,
    pub bits_per_sample: i32,
    pub sample_format: SampleFormat,
    pub has_loop: bool,
    pub loop_start_frames: i64,
    pub loop_end_frames: i64,
    pub loop_mode: SampleLoopMode,
    pub has_sustain_loop: bool,
    pub sustain_loop_start_frames: i64,
    pub sustain_loop_end_frames: i64,
    pub sustain_loop_mode: SampleLoopMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleLoopMode {
    None,
    Forward,
    PingPong,
}

impl SampleLoopMode {
    fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Forward,
            2 => Self::PingPong,
            _ => Self::None,
        }
    }

    pub fn to_raw(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Forward => 1,
            Self::PingPong => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleLoopRegion {
    pub start_frames: i64,
    pub end_frames: i64,
    pub mode: SampleLoopMode,
}

impl SampleLoopRegion {
    pub const fn none() -> Self {
        Self {
            start_frames: 0,
            end_frames: 0,
            mode: SampleLoopMode::None,
        }
    }

    pub const fn forward(start_frames: i64, end_frames: i64) -> Self {
        Self {
            start_frames,
            end_frames,
            mode: SampleLoopMode::Forward,
        }
    }

    pub const fn ping_pong(start_frames: i64, end_frames: i64) -> Self {
        Self {
            start_frames,
            end_frames,
            mode: SampleLoopMode::PingPong,
        }
    }

    pub fn has_loop(self) -> bool {
        self.mode != SampleLoopMode::None && self.end_frames > self.start_frames
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleLoopInfo {
    pub normal: SampleLoopRegion,
    pub sustain: SampleLoopRegion,
}

impl SampleLoopInfo {
    pub const fn none() -> Self {
        Self {
            normal: SampleLoopRegion::none(),
            sustain: SampleLoopRegion::none(),
        }
    }

    pub const fn forward(start_frames: i64, end_frames: i64) -> Self {
        Self {
            normal: SampleLoopRegion::forward(start_frames, end_frames),
            sustain: SampleLoopRegion::none(),
        }
    }

    #[allow(dead_code)]
    pub const fn ping_pong(start_frames: i64, end_frames: i64) -> Self {
        Self {
            normal: SampleLoopRegion::ping_pong(start_frames, end_frames),
            sustain: SampleLoopRegion::none(),
        }
    }

    pub const fn with_loops(normal: SampleLoopRegion, sustain: SampleLoopRegion) -> Self {
        Self { normal, sustain }
    }

    pub fn has_normal_loop(self) -> bool {
        self.normal.has_loop()
    }

    pub fn has_sustain_loop(self) -> bool {
        self.sustain.has_loop()
    }

    #[allow(dead_code)]
    pub fn pre_keyoff_loop(self) -> SampleLoopRegion {
        if self.has_sustain_loop() {
            self.sustain
        } else {
            self.normal
        }
    }

    #[allow(dead_code)]
    pub fn post_keyoff_loop(self) -> SampleLoopRegion {
        if self.has_normal_loop() {
            self.normal
        } else {
            SampleLoopRegion::none()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModuleMetadata {
    pub title: String,
    pub artist: String,
    pub tracker: String,
    pub type_long: String,
    pub date: String,
    pub message: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModuleInfo {
    pub title: String,
    pub tracker: String,
    pub format_type: String,
    pub format_type_long: String,
    pub artist: String,
    pub date: String,
    pub message: String,
    pub num_channels: i32,
    pub num_orders: i32,
    pub num_patterns: i32,
    pub num_instruments: i32,
    pub samples: Vec<SampleInfo>,
    pub file_size_bytes: u64,
}

/// Safe wrapper around an openmpt_module.
/// Not Send/Sync — must be used from a single thread.
pub struct Module {
    ptr: *mut OpenmptModule,
}

#[allow(dead_code)]
impl Module {
    /// Load a module from file data in memory.
    pub fn from_memory(data: &[u8]) -> Result<Self, String> {
        let mut error: c_int = 0;
        let mut error_message: *const c_char = std::ptr::null();
        let ptr = unsafe {
            openmpt_module_create_from_memory2(
                data.as_ptr() as *const c_void,
                data.len(),
                None,
                std::ptr::null_mut(),
                None,
                std::ptr::null_mut(),
                &mut error,
                &mut error_message,
                std::ptr::null(),
            )
        };
        if ptr.is_null() {
            let msg = unsafe { openmpt_string_to_rust(error_message) }
                .unwrap_or_else(|| format!("Unknown error (code {error})"));
            return Err(msg);
        }
        Ok(Module { ptr })
    }

    fn get_metadata_str(&self, key: &str) -> String {
        let Ok(c_key) = CString::new(key) else {
            return String::new();
        };
        unsafe { openmpt_string_to_rust(openmpt_module_get_metadata(self.ptr, c_key.as_ptr())) }
            .unwrap_or_default()
    }

    fn sample_format_from_raw(raw: i32) -> SampleFormat {
        match raw {
            SAMPLE_FORMAT_INT8 => SampleFormat::Int8,
            SAMPLE_FORMAT_INT16 => SampleFormat::Int16,
            SAMPLE_FORMAT_FLOAT32 => SampleFormat::Float32,
            SAMPLE_FORMAT_FLOAT64 => SampleFormat::Float64,
            _ => SampleFormat::Int8,
        }
    }

    fn sample_loop_info_from_raw(
        loop_start_frames: i64,
        loop_end_frames: i64,
        loop_mode_raw: i32,
        sustain_loop_start_frames: i64,
        sustain_loop_end_frames: i64,
        sustain_loop_mode_raw: i32,
    ) -> SampleLoopInfo {
        SampleLoopInfo::with_loops(
            SampleLoopRegion {
                start_frames: loop_start_frames,
                end_frames: loop_end_frames,
                mode: SampleLoopMode::from_raw(loop_mode_raw),
            },
            SampleLoopRegion {
                start_frames: sustain_loop_start_frames,
                end_frames: sustain_loop_end_frames,
                mode: SampleLoopMode::from_raw(sustain_loop_mode_raw),
            },
        )
    }

    /// Get full module info.
    pub fn info(&self) -> ModuleInfo {
        let num_samples = unsafe { openmpt_module_get_num_samples(self.ptr) };
        let mut samples = Vec::with_capacity(num_samples as usize);
        for i in 0..num_samples {
            let name =
                unsafe { openmpt_string_to_rust(openmpt_module_get_sample_name(self.ptr, i)) }
                    .unwrap_or_default();
            let rate = unsafe { openmpt_module_get_sample_rate(self.ptr, i) };
            let length_frames = unsafe { openmpt_module_get_sample_length_frames(self.ptr, i) };
            let channels = unsafe { openmpt_module_get_sample_channels(self.ptr, i) };
            let bits_per_sample = unsafe { openmpt_module_get_sample_bits_per_sample(self.ptr, i) };
            let sample_format = Self::sample_format_from_raw(unsafe {
                openmpt_module_get_sample_format(self.ptr, i)
            });
            let loop_info = Self::sample_loop_info_from_raw(
                unsafe { openmpt_module_get_sample_loop_start(self.ptr, i) },
                unsafe { openmpt_module_get_sample_loop_end(self.ptr, i) },
                unsafe { openmpt_module_get_sample_loop_mode(self.ptr, i) },
                unsafe { openmpt_module_get_sample_sustain_loop_start(self.ptr, i) },
                unsafe { openmpt_module_get_sample_sustain_loop_end(self.ptr, i) },
                unsafe { openmpt_module_get_sample_sustain_loop_mode(self.ptr, i) },
            );
            samples.push(SampleInfo {
                index: i,
                name,
                rate,
                length_frames,
                channels,
                bits_per_sample,
                sample_format,
                has_loop: loop_info.has_normal_loop(),
                loop_start_frames: loop_info.normal.start_frames,
                loop_end_frames: loop_info.normal.end_frames,
                loop_mode: loop_info.normal.mode,
                has_sustain_loop: loop_info.has_sustain_loop(),
                sustain_loop_start_frames: loop_info.sustain.start_frames,
                sustain_loop_end_frames: loop_info.sustain.end_frames,
                sustain_loop_mode: loop_info.sustain.mode,
            });
        }

        ModuleInfo {
            title: self.get_metadata_str("title"),
            tracker: self.get_metadata_str("tracker"),
            format_type: self.get_metadata_str("type"),
            format_type_long: self.get_metadata_str("type_long"),
            artist: self.get_metadata_str("artist"),
            date: self.get_metadata_str("date"),
            message: self.get_metadata_str("message"),
            num_channels: unsafe { openmpt_module_get_num_channels(self.ptr) },
            num_orders: unsafe { openmpt_module_get_num_orders(self.ptr) },
            num_patterns: unsafe { openmpt_module_get_num_patterns(self.ptr) },
            num_instruments: unsafe { openmpt_module_get_num_instruments(self.ptr) },
            samples,
            file_size_bytes: 0, // set by caller who knows the file size
        }
    }

    /// Render interleaved stereo float audio.
    /// Returns the number of frames actually rendered (0 = end of song).
    pub fn read_interleaved_float_stereo(&mut self, samplerate: i32, buffer: &mut [f32]) -> usize {
        let count = buffer.len() / 2;
        unsafe {
            openmpt_module_read_interleaved_float_stereo(
                self.ptr,
                samplerate,
                count,
                buffer.as_mut_ptr(),
            )
        }
    }

    /// Render interleaved stereo double-precision audio.
    /// Returns the number of frames actually rendered (0 = end of song).
    pub fn read_interleaved_double_stereo(&mut self, samplerate: i32, buffer: &mut [f64]) -> usize {
        let count = buffer.len() / 2;
        unsafe {
            openmpt_module_read_interleaved_double_stereo(
                self.ptr,
                samplerate,
                count,
                buffer.as_mut_ptr(),
            )
        }
    }

    #[cfg(test)]
    pub fn last_error_code(&self) -> i32 {
        unsafe { openmpt_module_error_get_last(self.ptr) }
    }

    #[cfg(test)]
    pub fn last_error_message(&self) -> String {
        unsafe { openmpt_string_to_rust(openmpt_module_error_get_last_message(self.ptr)) }
            .unwrap_or_default()
    }

    // Playback control

    pub fn position_seconds(&self) -> f64 {
        unsafe { openmpt_module_get_position_seconds(self.ptr) }
    }

    pub fn set_position_seconds(&mut self, seconds: f64) -> f64 {
        unsafe { openmpt_module_set_position_seconds(self.ptr, seconds) }
    }

    pub fn duration_seconds(&self) -> f64 {
        unsafe { openmpt_module_get_duration_seconds(self.ptr) }
    }

    pub fn current_order(&self) -> i32 {
        unsafe { openmpt_module_get_current_order(self.ptr) }
    }

    pub fn current_row(&self) -> i32 {
        unsafe { openmpt_module_get_current_row(self.ptr) }
    }

    pub fn current_pattern(&self) -> i32 {
        unsafe { openmpt_module_get_current_pattern(self.ptr) }
    }

    pub fn set_repeat_count(&mut self, count: i32) {
        unsafe {
            openmpt_module_set_repeat_count(self.ptr, count);
        }
    }

    pub fn set_interpolation_filter(&mut self, length: i32) {
        unsafe {
            openmpt_module_set_render_param(self.ptr, RENDER_INTERPOLATIONFILTER_LENGTH, length);
        }
    }

    pub fn set_aniso64_k_beta(&mut self, value: f64) {
        let ctl = c"render.resampler.aniso64_k_beta";
        unsafe {
            openmpt_module_ctl_set_floatingpoint(self.ptr, ctl.as_ptr(), value);
        }
    }

    pub fn set_aniso64_k_beta2(&mut self, value: f64) {
        let ctl = c"render.resampler.aniso64_k_beta2";
        unsafe {
            openmpt_module_ctl_set_floatingpoint(self.ptr, ctl.as_ptr(), value);
        }
    }

    pub fn aniso64_k_beta(&self) -> f64 {
        let ctl = c"render.resampler.aniso64_k_beta";
        unsafe { openmpt_module_ctl_get_floatingpoint(self.ptr, ctl.as_ptr()) }
    }

    pub fn aniso64_k_beta2(&self) -> f64 {
        let ctl = c"render.resampler.aniso64_k_beta2";
        unsafe { openmpt_module_ctl_get_floatingpoint(self.ptr, ctl.as_ptr()) }
    }

    pub fn set_stereo_separation(&mut self, percent: i32) {
        unsafe {
            openmpt_module_set_render_param(self.ptr, RENDER_STEREOSEPARATION_PERCENT, percent);
        }
    }

    pub fn stereo_separation(&self) -> Option<i32> {
        self.render_param(RENDER_STEREOSEPARATION_PERCENT)
    }

    pub fn set_volume_ramping_strength(&mut self, strength: i32) {
        unsafe {
            openmpt_module_set_render_param(self.ptr, RENDER_VOLUMERAMPING_STRENGTH, strength);
        }
    }

    pub fn volume_ramping_strength(&self) -> Option<i32> {
        self.render_param(RENDER_VOLUMERAMPING_STRENGTH)
    }

    pub fn agc_enabled(&self) -> bool {
        unsafe { openmpt_module_get_agc_enabled(self.ptr) != 0 }
    }

    pub fn set_agc_enabled(&mut self, enabled: bool) -> bool {
        unsafe { openmpt_module_set_agc_enabled(self.ptr, enabled as c_int) != 0 }
    }

    pub fn agc_profile(&self) -> AgcProfile {
        match unsafe { openmpt_module_get_agc_profile(self.ptr) } {
            AGC_PROFILE_GENTLE => AgcProfile::Gentle,
            _ => AgcProfile::Stock,
        }
    }

    pub fn set_agc_profile(&mut self, profile: AgcProfile) -> bool {
        let profile = match profile {
            AgcProfile::Stock => AGC_PROFILE_STOCK,
            AgcProfile::Gentle => AGC_PROFILE_GENTLE,
        };
        unsafe { openmpt_module_set_agc_profile(self.ptr, profile) != 0 }
    }

    pub fn apply_quinlight_processing_settings(
        &mut self,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
    ) {
        self.set_interpolation_filter(interpolation_filter);
        self.set_stereo_separation(stereo_separation);
        self.set_volume_ramping_strength(DEFAULT_VOLUMERAMPING_STRENGTH);
        self.set_agc_profile(DEFAULT_AGC_PROFILE);
        self.set_agc_enabled(agc_enabled);
    }

    pub fn set_master_gain_millibel(&mut self, millibel: i32) {
        unsafe {
            openmpt_module_set_render_param(self.ptr, RENDER_MASTERGAIN_MILLIBEL, millibel);
        }
    }

    #[cfg(test)]
    pub fn set_test_preamp(&mut self, preamp: i32) -> bool {
        unsafe { openmpt_module_set_test_preamp(self.ptr, preamp) != 0 }
    }

    pub fn interpolation_filter(&self) -> Option<i32> {
        self.render_param(RENDER_INTERPOLATIONFILTER_LENGTH)
    }

    fn render_param(&self, param: c_int) -> Option<i32> {
        let mut value = 0;
        let ok = unsafe { openmpt_module_get_render_param(self.ptr, param, &mut value) };
        (ok != 0).then_some(value)
    }

    pub fn num_channels(&self) -> i32 {
        unsafe { openmpt_module_get_num_channels(self.ptr) }
    }

    /// Get per-channel VU levels (left, right) for all channels.
    pub fn channel_vu(&self) -> Vec<(f64, f64)> {
        let n = self.num_channels();
        (0..n)
            .map(|ch| unsafe {
                (
                    openmpt_module_get_current_channel_vu_left(self.ptr, ch),
                    openmpt_module_get_current_channel_vu_right(self.ptr, ch),
                )
            })
            .collect()
    }

    /// Get the set of sample indices currently playing across all channels.
    pub fn active_samples(&self) -> Vec<i32> {
        let n = self.num_channels();
        let mut active = Vec::new();
        for ch in 0..n {
            let idx = unsafe { openmpt_module_get_current_channel_sample(self.ptr, ch) };
            if idx >= 0 && !active.contains(&idx) {
                active.push(idx);
            }
        }
        active
    }

    pub fn has_sample_loop(&self, index: i32) -> bool {
        unsafe { openmpt_module_has_sample_loop(self.ptr, index) != 0 }
    }

    pub fn sample_loop_info(&self, index: i32) -> SampleLoopInfo {
        Self::sample_loop_info_from_raw(
            unsafe { openmpt_module_get_sample_loop_start(self.ptr, index) },
            unsafe { openmpt_module_get_sample_loop_end(self.ptr, index) },
            unsafe { openmpt_module_get_sample_loop_mode(self.ptr, index) },
            unsafe { openmpt_module_get_sample_sustain_loop_start(self.ptr, index) },
            unsafe { openmpt_module_get_sample_sustain_loop_end(self.ptr, index) },
            unsafe { openmpt_module_get_sample_sustain_loop_mode(self.ptr, index) },
        )
    }

    pub fn num_patterns(&self) -> i32 {
        unsafe { openmpt_module_get_num_patterns(self.ptr) }
    }

    #[allow(dead_code)] // used in tests
    pub fn get_order_pattern(&self, order: i32) -> i32 {
        unsafe { openmpt_module_get_order_pattern(self.ptr, order) }
    }

    pub fn pattern_num_rows(&self, pattern: i32) -> i32 {
        unsafe { openmpt_module_get_pattern_num_rows(self.ptr, pattern) }
    }

    pub fn pattern_rows_per_beat(&self, pattern: i32) -> i32 {
        unsafe { openmpt_module_get_pattern_rows_per_beat(self.ptr, pattern) }
    }

    pub fn get_pattern_command(&self, pattern: i32, row: i32, channel: i32, cmd: i32) -> u8 {
        unsafe {
            openmpt_module_get_pattern_row_channel_command(self.ptr, pattern, row, channel, cmd)
        }
    }

    /// Get a pre-formatted pattern row for a single channel (e.g. "C-5 01 v64 S30").
    pub fn format_pattern_row_channel(&self, pattern: i32, row: i32, channel: i32) -> String {
        unsafe {
            openmpt_string_to_rust(openmpt_module_format_pattern_row_channel(
                self.ptr, pattern, row, channel, 0, 1,
            ))
        }
        .unwrap_or_default()
    }

    /// Get highlight codes for a pattern row channel (parallel to format string).
    pub fn highlight_pattern_row_channel(&self, pattern: i32, row: i32, channel: i32) -> String {
        unsafe {
            openmpt_string_to_rust(openmpt_module_highlight_pattern_row_channel(
                self.ptr, pattern, row, channel, 0, 1,
            ))
        }
        .unwrap_or_default()
    }

    pub fn current_bpm(&self) -> f64 {
        unsafe { openmpt_module_get_current_estimated_bpm(self.ptr) }
    }

    pub fn current_tempo2(&self) -> f64 {
        unsafe { openmpt_module_get_current_tempo2(self.ptr) }
    }

    pub fn current_speed(&self) -> i32 {
        unsafe { openmpt_module_get_current_speed(self.ptr) }
    }

    pub fn channel_name(&self, index: i32) -> String {
        unsafe { openmpt_string_to_rust(openmpt_module_get_channel_name(self.ptr, index)) }
            .unwrap_or_default()
    }

    pub fn set_pattern_command(
        &mut self,
        pattern: i32,
        row: i32,
        channel: i32,
        cmd: i32,
        value: u8,
    ) -> bool {
        unsafe {
            openmpt_module_set_pattern_row_channel_command(
                self.ptr, pattern, row, channel, cmd, value,
            ) != 0
        }
    }

    // Sample data access (quinlight extensions)

    pub fn num_samples(&self) -> i32 {
        unsafe { openmpt_module_get_num_samples(self.ptr) }
    }

    pub fn sample_name(&self, index: i32) -> String {
        unsafe { openmpt_string_to_rust(openmpt_module_get_sample_name(self.ptr, index)) }
            .unwrap_or_default()
    }

    pub fn num_instruments(&self) -> i32 {
        unsafe { openmpt_module_get_num_instruments(self.ptr) }
    }

    pub fn instrument_name(&self, index: i32) -> String {
        unsafe { openmpt_string_to_rust(openmpt_module_get_instrument_name(self.ptr, index)) }
            .unwrap_or_default()
    }

    pub fn metadata(&self) -> ModuleMetadata {
        ModuleMetadata {
            title: self.get_metadata_str("title"),
            artist: self.get_metadata_str("artist"),
            tracker: self.get_metadata_str("tracker"),
            type_long: self.get_metadata_str("type_long"),
            date: self.get_metadata_str("date"),
            message: self.get_metadata_str("message"),
        }
    }

    pub fn sample_rate(&self, index: i32) -> i32 {
        unsafe { openmpt_module_get_sample_rate(self.ptr, index) }
    }

    pub fn sample_c5_speed(&self, index: i32) -> i32 {
        unsafe { openmpt_module_get_sample_c5_speed(self.ptr, index) }
    }

    pub fn sample_relative_tone(&self, index: i32) -> i8 {
        unsafe { openmpt_module_get_sample_relative_tone(self.ptr, index) as i8 }
    }

    pub fn sample_fine_tune(&self, index: i32) -> i8 {
        unsafe { openmpt_module_get_sample_fine_tune(self.ptr, index) as i8 }
    }

    pub fn sample_default_volume(&self, index: i32) -> u16 {
        unsafe { openmpt_module_get_sample_default_volume(self.ptr, index).max(0) as u16 }
    }

    pub fn sample_default_pan(&self, index: i32) -> Option<u16> {
        let has_pan = unsafe { openmpt_module_has_sample_default_pan(self.ptr, index) != 0 };
        has_pan.then(|| unsafe {
            openmpt_module_get_sample_default_pan(self.ptr, index).max(0) as u16
        })
    }

    pub fn sample_length_frames(&self, index: i32) -> i64 {
        unsafe { openmpt_module_get_sample_length_frames(self.ptr, index) }
    }

    pub fn sample_channels(&self, index: i32) -> i32 {
        unsafe { openmpt_module_get_sample_channels(self.ptr, index) }
    }

    pub fn sample_bits_per_sample(&self, index: i32) -> i32 {
        unsafe { openmpt_module_get_sample_bits_per_sample(self.ptr, index) }
    }

    pub fn sample_format(&self, index: i32) -> SampleFormat {
        Self::sample_format_from_raw(unsafe { openmpt_module_get_sample_format(self.ptr, index) })
    }

    pub fn instrument_sample_for_note(&self, instrument_index: i32, note: u8) -> Option<i32> {
        let sample = unsafe {
            openmpt_module_get_instrument_keyboard_sample(self.ptr, instrument_index, note as i32)
        };
        (sample >= 0).then_some(sample)
    }

    /// Read sample data as f64. Returns the data as Vec<f64> (interleaved if stereo).
    pub fn read_sample_data(&self, index: i32) -> Option<Vec<f64>> {
        let length = self.sample_length_frames(index);
        let channels = self.sample_channels(index);
        if length <= 0 || channels <= 0 {
            return None;
        }
        let total = (length as usize) * (channels as usize);
        let mut buffer = vec![0.0f64; total];
        let read = unsafe {
            openmpt_module_read_sample_data(self.ptr, index, buffer.as_mut_ptr(), length)
        };
        if read <= 0 {
            return None;
        }
        buffer.truncate((read as usize) * (channels as usize));
        Some(buffer)
    }

    /// Replace a sample's data with f64 data. Stored as Float64 internally.
    pub fn replace_sample_data(
        &mut self,
        index: i32,
        data: &[f64],
        length_frames: i64,
        channels: i32,
        new_sample_rate: i32,
    ) -> bool {
        let result = unsafe {
            openmpt_module_replace_sample_data(
                self.ptr,
                index,
                data.as_ptr(),
                length_frames,
                channels,
                new_sample_rate,
            )
        };
        result != 0
    }

    /// Replace a sample's data without scaling loop points.
    /// Caller must set loop points explicitly via `set_sample_loop_points` afterwards.
    pub fn replace_sample_data_raw(
        &mut self,
        index: i32,
        data: &[f64],
        length_frames: i64,
        channels: i32,
        new_sample_rate: i32,
    ) -> bool {
        let result = unsafe {
            openmpt_module_replace_sample_data_raw(
                self.ptr,
                index,
                data.as_ptr(),
                length_frames,
                channels,
                new_sample_rate,
            )
        };
        result != 0
    }

    /// Set explicit loop points on a sample.
    /// Loop mode: 0 = none, 1 = forward, 2 = ping-pong.
    pub fn set_sample_loop_points(&mut self, index: i32, loop_info: &SampleLoopInfo) -> bool {
        let result = unsafe {
            openmpt_module_set_sample_loop_points(
                self.ptr,
                index,
                loop_info.normal.start_frames,
                loop_info.normal.end_frames,
                loop_info.normal.mode.to_raw(),
                loop_info.sustain.start_frames,
                loop_info.sustain.end_frames,
                loop_info.sustain.mode.to_raw(),
            )
        };
        result != 0
    }

    /// Reset per-channel mixer state tied to the given sample's rate without
    /// touching position, envelopes, or LFO phases. Intended to be called
    /// right after `replace_sample_data` / `_raw` when the rate changed — it
    /// clears the filter-memory / resampler-history transients that otherwise
    /// bleed from the old rate into the new.
    pub fn refresh_channels_for_sample(&mut self, index: i32) -> bool {
        unsafe { openmpt_module_refresh_channels_for_sample(self.ptr, index) != 0 }
    }

    pub fn linear_slides_enabled(&self) -> bool {
        unsafe { openmpt_module_get_linear_slides(self.ptr) != 0 }
    }

    pub fn set_linear_slides(&mut self, enabled: bool) -> bool {
        unsafe { openmpt_module_set_linear_slides(self.ptr, enabled as c_int) != 0 }
    }

    /// Save the module (with modified samples) to a byte vector.
    /// Returns None if the save fails.
    #[allow(dead_code)] // used in tests
    pub fn save_to_memory(&self) -> Option<Vec<u8>> {
        // First call with null to get required size
        let size = unsafe { openmpt_module_save_to_memory(self.ptr, std::ptr::null_mut(), 0) };
        if size <= 0 {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        let written = unsafe { openmpt_module_save_to_memory(self.ptr, buf.as_mut_ptr(), size) };
        if written <= 0 {
            return None;
        }
        buf.truncate(written as usize);
        Some(buf)
    }

    pub fn save_loaded_format_to_memory(&self) -> Option<Vec<u8>> {
        let size = unsafe {
            openmpt_module_save_loaded_format_to_memory(self.ptr, std::ptr::null_mut(), 0)
        };
        if size <= 0 {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        let written = unsafe {
            openmpt_module_save_loaded_format_to_memory(self.ptr, buf.as_mut_ptr(), size)
        };
        if written <= 0 {
            return None;
        }
        buf.truncate(written as usize);
        Some(buf)
    }

    pub fn save_best_format_to_memory(&self) -> Option<Vec<u8>> {
        let size =
            unsafe { openmpt_module_save_best_format_to_memory(self.ptr, std::ptr::null_mut(), 0) };
        if size <= 0 {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        let written =
            unsafe { openmpt_module_save_best_format_to_memory(self.ptr, buf.as_mut_ptr(), size) };
        if written <= 0 {
            return None;
        }
        buf.truncate(written as usize);
        Some(buf)
    }

    pub fn loaded_format_extension(&self) -> String {
        unsafe { openmpt_string_to_rust(openmpt_module_get_loaded_format_extension(self.ptr)) }
            .unwrap_or_default()
    }

    pub fn best_save_format_extension(&self) -> String {
        unsafe { openmpt_string_to_rust(openmpt_module_get_best_save_format_extension(self.ptr)) }
            .unwrap_or_default()
    }

    pub(crate) fn note_from_period(&self, period: f64, fine_tune: i32, c5speed: f64) -> f64 {
        unsafe {
            openmpt_module_quinlight_test_get_note_from_period(self.ptr, period, fine_tune, c5speed)
        }
    }

    pub(crate) fn period_from_note(&self, note: u32, fine_tune: i32, c5speed: f64) -> f64 {
        unsafe {
            openmpt_module_quinlight_test_get_period_from_note(self.ptr, note, fine_tune, c5speed)
        }
    }

    pub(crate) fn frequency_from_period(&self, period: f64, c5speed: f64) -> f64 {
        unsafe { openmpt_module_quinlight_get_freq_from_period(self.ptr, period, c5speed) }
    }

    pub(crate) fn current_channel_period(&self, channel: i32) -> f64 {
        unsafe { openmpt_module_quinlight_test_get_current_channel_period(self.ptr, channel) }
    }

    pub(crate) fn current_channel_frequency(&self, channel: i32) -> f64 {
        unsafe { openmpt_module_quinlight_test_get_current_channel_frequency(self.ptr, channel) }
    }

    pub(crate) fn current_channel_increment(&self, channel: i32) -> f64 {
        unsafe { openmpt_module_quinlight_test_get_current_channel_increment(self.ptr, channel) }
    }

    #[cfg(test)]
    pub(crate) fn test_get_note_from_period(
        &self,
        period: f64,
        fine_tune: i32,
        c5speed: f64,
    ) -> f64 {
        self.note_from_period(period, fine_tune, c5speed)
    }

    #[cfg(test)]
    pub(crate) fn test_get_period_from_note(&self, note: u32, fine_tune: i32, c5speed: f64) -> f64 {
        self.period_from_note(note, fine_tune, c5speed)
    }

    #[cfg(test)]
    pub(crate) fn test_get_current_channel_period(&self, channel: i32) -> f64 {
        self.current_channel_period(channel)
    }

    #[cfg(test)]
    pub(crate) fn test_get_current_channel_frequency(&self, channel: i32) -> f64 {
        self.current_channel_frequency(channel)
    }

    #[cfg(test)]
    pub(crate) fn test_get_current_channel_increment(&self, channel: i32) -> f64 {
        self.current_channel_increment(channel)
    }

    #[cfg(test)]
    pub(crate) fn test_apply_linear_pitch_slide(
        target: f64,
        total_amount: i32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_apply_linear_pitch_slide(
                target,
                total_amount,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_apply_continuous_linear_pitch_slide(
        target: f64,
        total_amount: i32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_apply_continuous_linear_pitch_slide(
                target,
                total_amount,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_apply_it_linear_pitch_slide_reference(
        target: f64,
        total_amount: i32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_apply_it_linear_pitch_slide_reference(
                target,
                total_amount,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_pitch_envelope_factor(envval: i32, periods_are_frequencies: bool) -> f64 {
        unsafe {
            openmpt_quinlight_test_pitch_envelope_factor(envval, periods_are_frequencies as c_int)
        }
    }

    #[cfg(test)]
    pub(crate) fn test_pitch_envelope_reference_factor(
        envval: i32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_pitch_envelope_reference_factor(
                envval,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_it_arpeggio_factor(semitones: u32, periods_are_frequencies: bool) -> f64 {
        unsafe {
            openmpt_quinlight_test_it_arpeggio_factor(semitones, periods_are_frequencies as c_int)
        }
    }

    #[cfg(test)]
    pub(crate) fn test_it_arpeggio_reference_factor(
        semitones: u32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_it_arpeggio_reference_factor(
                semitones,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_it_autovibrato_factor(vdelta: i32, periods_are_frequencies: bool) -> f64 {
        unsafe {
            openmpt_quinlight_test_it_autovibrato_factor(vdelta, periods_are_frequencies as c_int)
        }
    }

    #[cfg(test)]
    pub(crate) fn test_it_autovibrato_reference_factor(
        vdelta: i32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_it_autovibrato_reference_factor(
                vdelta,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_linear_autovibrato_factor(n: i32, periods_are_frequencies: bool) -> f64 {
        unsafe {
            openmpt_quinlight_test_linear_autovibrato_factor(n, periods_are_frequencies as c_int)
        }
    }

    #[cfg(test)]
    pub(crate) fn test_linear_autovibrato_reference_factor(
        n: i32,
        periods_are_frequencies: bool,
    ) -> f64 {
        unsafe {
            openmpt_quinlight_test_linear_autovibrato_reference_factor(
                n,
                periods_are_frequencies as c_int,
            )
        }
    }

    #[cfg(test)]
    pub(crate) fn test_microtuning_factor(finetune: i32) -> f64 {
        unsafe { openmpt_quinlight_test_microtuning_factor(finetune) }
    }

    #[cfg(test)]
    pub(crate) fn test_hertz_from_note(note: u32, c5speed: f64) -> f64 {
        unsafe { openmpt_quinlight_test_hertz_from_note(note, c5speed) }
    }

    #[cfg(test)]
    pub(crate) fn test_reference_hertz_from_note(note: u32, c5speed: f64) -> f64 {
        unsafe { openmpt_quinlight_test_reference_hertz_from_note(note, c5speed) }
    }

    #[cfg(test)]
    pub(crate) fn test_xm_linear_freq_from_period(period: u32) -> f64 {
        unsafe { openmpt_quinlight_test_xm_linear_freq_from_period(period) }
    }

    #[cfg(test)]
    pub(crate) fn test_reference_xm_linear_freq_from_period(period: u32) -> f64 {
        unsafe { openmpt_quinlight_test_reference_xm_linear_freq_from_period(period) }
    }
}

// SAFETY: Module wraps a *mut OpenmptModule. libopenmpt documents that
// consecutive accesses from different threads are fine (just not concurrent).
// We protect concurrent access with a Mutex in player.rs.
unsafe impl Send for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { openmpt_module_destroy(self.ptr) };
        }
    }
}
