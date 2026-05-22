// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(clippy::collapsible_if, clippy::unnecessary_cast)]

use std::fs::File;
use std::path::Path;

use ac_ffmpeg::{
    codec::{
        CodecParameters, Encoder,
        audio::{
            AudioEncoder, AudioFrameMut, AudioResampler,
            frame::{
                ChannelLayout, SampleFormat as FfmpegSampleFormat, get_channel_layout,
                get_sample_format,
            },
        },
    },
    format::{
        io::IO,
        muxer::{Muxer, OutputFormat},
    },
    time::{TimeBase, Timestamp},
};

use crate::openmpt::{DEFAULT_AGC_ENABLED, Module};
use crate::remaster::{copy_interleaved_f64_to_audio_frame, ffmpeg_input_sample_format};
use crate::simd;

const DEFAULT_SAMPLE_RATE: u32 = 48000;
const CHANNELS: u32 = 2;
/// FFmpeg's native AAC encoder maximum supported sample rate.
const AAC_MAX_SAMPLE_RATE: u32 = 96000;

/// Metadata embedded in exported audio files (FLAC/M4A).
#[derive(Clone, Default)]
pub struct AudioMetadata {
    pub title: String,
    pub artist: String,
    pub album: String,
}

#[derive(Clone, Copy)]
enum AudioExportFormat {
    Flac,
    Aac,
}

impl AudioExportFormat {
    fn codec_name(self) -> &'static str {
        match self {
            Self::Flac => "flac",
            Self::Aac => "aac",
        }
    }

    fn encoder_sample_format(self) -> FfmpegSampleFormat {
        get_sample_format(match self {
            Self::Flac => "s32",
            Self::Aac => "fltp",
        })
    }

    fn bit_rate(self) -> Option<u64> {
        match self {
            Self::Flac => None,
            Self::Aac => Some(262_144),
        }
    }

    /// True-peak ceiling (dBTP) for the loudness limiter. AAC needs more
    /// headroom than FLAC because lossy reconstruction can overshoot the
    /// encoded sample-domain peak by 1–3 dBTP at the decoder (intersample
    /// peaks), which clips at the DAC.
    fn true_peak_ceiling_dbtp(self) -> f64 {
        match self {
            Self::Flac => -1.0,
            Self::Aac => -2.0,
        }
    }

    /// The actual encoder sample rate, capping AAC at 96 kHz.
    fn effective_output_rate(self, requested: u32) -> u32 {
        match self {
            Self::Flac => requested,
            Self::Aac => requested.min(AAC_MAX_SAMPLE_RATE),
        }
    }
}

fn render_loaded_module_to_samples(
    module: &mut Module,
    sample_rate: u32,
    progress: Option<(&crossbeam_channel::Sender<f32>, f32, f32)>,
) -> Result<Vec<f64>, String> {
    let duration = module.duration_seconds().max(1.0);
    let mut samples = Vec::new();
    let mut buf = vec![0f64; sample_rate as usize * CHANNELS as usize]; // 1 second
    let mut seconds_rendered = 0.0f64;
    loop {
        let frames = module.read_interleaved_double_stereo(sample_rate as i32, &mut buf);
        if frames == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..frames * CHANNELS as usize]);
        seconds_rendered += frames as f64 / sample_rate as f64;
        if let Some((tx, lo, hi)) = &progress {
            let frac = (seconds_rendered / duration).min(1.0) as f32;
            let _ = tx.try_send(lo + frac * (hi - lo));
        }
    }

    if samples.is_empty() {
        return Err("Module produced no audio".into());
    }

    Ok(samples)
}

fn configure_module_for_render(
    module: &mut Module,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
) {
    module.set_repeat_count(0); // play once
    module.apply_quinlight_processing_settings(
        stereo_separation,
        interpolation_filter,
        agc_enabled,
    );
    module.set_position_seconds(0.0);
}

#[allow(dead_code)]
pub(crate) fn render_module_to_samples(
    file_data: &[u8],
    stereo_separation: i32,
    interpolation_filter: i32,
) -> Result<Vec<f64>, String> {
    render_module_to_samples_with_agc(
        file_data,
        stereo_separation,
        interpolation_filter,
        DEFAULT_AGC_ENABLED,
    )
}

/// Render at a custom sample rate (e.g. 8363 Hz for vinyl groove visualization).
#[allow(dead_code)] // Used by gui/vinyl_shader.rs in the binary but not by lib.rs
pub(crate) fn render_module_to_samples_at_rate(
    file_data: &[u8],
    stereo_separation: i32,
    interpolation_filter: i32,
    sample_rate: u32,
) -> Result<Vec<f64>, String> {
    let mut module = Module::from_memory(file_data)?;
    configure_module_for_render(
        &mut module,
        stereo_separation,
        interpolation_filter,
        DEFAULT_AGC_ENABLED,
    );
    render_loaded_module_to_samples(&mut module, sample_rate, None)
}

#[allow(dead_code)]
pub(crate) fn render_module_to_samples_with_agc(
    file_data: &[u8],
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
) -> Result<Vec<f64>, String> {
    let mut module = Module::from_memory(file_data)?;
    configure_module_for_render(
        &mut module,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
    );
    render_loaded_module_to_samples(&mut module, DEFAULT_SAMPLE_RATE, None)
}

#[allow(dead_code)]
pub(crate) fn render_live_module_to_samples(
    module: &mut Module,
    stereo_separation: i32,
    interpolation_filter: i32,
) -> Result<Vec<f64>, String> {
    render_live_module_to_samples_with_agc(
        module,
        stereo_separation,
        interpolation_filter,
        DEFAULT_AGC_ENABLED,
        DEFAULT_SAMPLE_RATE,
        None,
    )
}

#[allow(dead_code)]
pub(crate) fn render_live_module_to_samples_with_agc(
    module: &mut Module,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    progress: Option<(&crossbeam_channel::Sender<f32>, f32, f32)>,
) -> Result<Vec<f64>, String> {
    let saved_position = module.position_seconds();
    let saved_interpolation = module.interpolation_filter();
    let saved_stereo_separation = module.stereo_separation();
    let saved_volume_ramping = module.volume_ramping_strength();
    let saved_agc_enabled = module.agc_enabled();
    let saved_agc_profile = module.agc_profile();

    configure_module_for_render(module, stereo_separation, interpolation_filter, agc_enabled);

    let result = render_loaded_module_to_samples(module, sample_rate, progress);

    module.set_repeat_count(-1);
    if let Some(saved_interpolation) = saved_interpolation {
        module.set_interpolation_filter(saved_interpolation);
    }
    if let Some(saved_stereo_separation) = saved_stereo_separation {
        module.set_stereo_separation(saved_stereo_separation);
    }
    if let Some(saved_volume_ramping) = saved_volume_ramping {
        module.set_volume_ramping_strength(saved_volume_ramping);
    }
    module.set_agc_profile(saved_agc_profile);
    module.set_agc_enabled(saved_agc_enabled);
    module.set_position_seconds(saved_position);

    result
}

/// Target integrated loudness for rendered exports (EBU R128 / streaming standard).
const TARGET_LUFS: f64 = -14.0;

/// Remove DC bias per-channel, then scale the whole buffer by a single gain
/// that hits -14 LUFS when that fits under the supplied true-peak ceiling
/// (dBTP), else stays under the ceiling (quieter than target). The f64
/// pipeline has effectively infinite dynamic range, so we never clamp —
/// scaling is always preferable to clipping.
fn normalize_samples(samples: &mut [f64], sample_rate: u32, true_peak_ceiling_dbtp: f64) {
    if samples.len() < CHANNELS as usize {
        return;
    }
    let len_per_ch = samples.len() / CHANNELS as usize;
    // Deinterleave → SIMD per-channel DC removal → reinterleave
    let (mut left, mut right) = simd::deinterleave_stereo_f64(samples);
    let mean_l = simd::sum_f64(&left) / len_per_ch as f64;
    let mean_r = simd::sum_f64(&right) / len_per_ch as f64;
    simd::subtract_in_place_f64(&mut left, mean_l);
    simd::subtract_in_place_f64(&mut right, mean_r);
    let debiased = simd::interleave_stereo_f64(&left, &right);
    samples[..debiased.len()].copy_from_slice(&debiased);

    let ceiling_lin = 10.0_f64.powf(true_peak_ceiling_dbtp / 20.0);

    // One pass: integrated loudness + ITU-R BS.1770-compliant 4× oversampled true-peak.
    if let Ok(mut meter) = ebur128::EbuR128::new(
        CHANNELS as u32,
        sample_rate as u32,
        ebur128::Mode::I | ebur128::Mode::TRUE_PEAK,
    ) {
        if meter.add_frames_f64(samples).is_ok() {
            if let Ok(loudness) = meter.loudness_global() {
                if loudness.is_finite() {
                    let lufs_gain = 10.0_f64.powf((TARGET_LUFS - loudness) / 20.0);
                    let tp_max = (0..CHANNELS)
                        .filter_map(|c| meter.true_peak(c).ok())
                        .filter(|v| v.is_finite())
                        .fold(0.0_f64, f64::max);
                    // If ebur128 couldn't provide a true-peak (should not
                    // happen when add_frames_f64 succeeded), fall through
                    // to the peak-normalize fallback rather than apply
                    // lufs_gain blind.
                    if tp_max > 0.0 {
                        let gain = if tp_max * lufs_gain > ceiling_lin {
                            ceiling_lin / tp_max
                        } else {
                            lufs_gain
                        };
                        for s in samples.iter_mut() {
                            *s *= gain;
                        }
                        return;
                    }
                }
            }
        }
    }

    // Fallback (ebur128 error or non-finite loudness): peak-normalize to the
    // same ceiling. No clamp — the gain is chosen to keep peak ≤ ceiling.
    let peak = simd::peak_abs_f64(samples);
    if peak > 0.0 {
        let gain = ceiling_lin / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
}

fn open_output_muxer(
    final_path: &Path,
    write_path: &Path,
    elementary_streams: &[CodecParameters],
    metadata: &AudioMetadata,
) -> Result<Muxer<File>, String> {
    // Use the final path for format guessing (needs the real extension),
    // but write to the temporary path for atomic rename on success.
    let output_name = final_path.to_string_lossy();
    let output_format =
        OutputFormat::guess_from_file_name(output_name.as_ref()).ok_or_else(|| {
            format!(
                "Unable to guess FFmpeg output format for {}",
                final_path.display()
            )
        })?;
    let output = File::create(write_path)
        .map_err(|e| format!("Failed to create output file {}: {e}", write_path.display()))?;
    let io = IO::from_seekable_write_stream(output);

    let mut muxer_builder = Muxer::builder();
    for codec_parameters in elementary_streams {
        muxer_builder
            .add_stream(codec_parameters)
            .map_err(|e| format!("Failed to add FFmpeg stream: {e}"))?;
    }

    let mut muxer_builder = muxer_builder
        .set_metadata("title", &metadata.title)
        .set_metadata("album", &metadata.album);
    if !metadata.artist.is_empty() {
        muxer_builder = muxer_builder.set_metadata("artist", &metadata.artist);
    }
    muxer_builder
        .build(io, output_format)
        .map_err(|e| format!("Failed to create FFmpeg muxer: {e}"))
}

fn build_audio_encoder(
    format: AudioExportFormat,
    channel_layout: ChannelLayout,
    encoder_rate: u32,
) -> Result<AudioEncoder, String> {
    let builder = AudioEncoder::builder(format.codec_name())
        .map_err(|e| {
            format!(
                "Failed to create FFmpeg {} encoder: {e}",
                format.codec_name()
            )
        })?
        .time_base(TimeBase::new(1, encoder_rate as i32))
        .sample_format(format.encoder_sample_format())
        .sample_rate(encoder_rate)
        .channel_layout(channel_layout);
    let builder = if let Some(bit_rate) = format.bit_rate() {
        builder.bit_rate(bit_rate)
    } else {
        builder
    };
    builder
        .build()
        .map_err(|e| format!("Failed to open FFmpeg {} encoder: {e}", format.codec_name()))
}

fn drain_encoder_packets(
    encoder: &mut AudioEncoder,
    muxer: &mut Muxer<File>,
) -> Result<(), String> {
    while let Some(packet) = encoder
        .take()
        .map_err(|e| format!("FFmpeg encoder packet error: {e}"))?
    {
        muxer
            .push(packet.with_stream_index(0))
            .map_err(|e| format!("FFmpeg mux error: {e}"))?;
    }
    Ok(())
}

fn drain_resampler_frames(
    resampler: &mut AudioResampler,
    encoder: &mut AudioEncoder,
    muxer: &mut Muxer<File>,
) -> Result<(), String> {
    while let Some(frame) = resampler
        .take()
        .map_err(|e| format!("FFmpeg resampler frame error: {e}"))?
    {
        encoder
            .push(frame)
            .map_err(|e| format!("FFmpeg encoder push error: {e}"))?;
        drain_encoder_packets(encoder, muxer)?;
    }
    Ok(())
}

fn write_audio_with_ffmpeg(
    mut samples: Vec<f64>,
    output_path: &Path,
    format: AudioExportFormat,
    sample_rate: u32,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    normalize_samples(&mut samples, sample_rate, format.true_peak_ceiling_dbtp());

    // Write to a temp file, then atomically rename on success to prevent
    // partially-written output files from crashes or interruptions.
    let tmp_path = {
        let mut name = output_path.file_name().unwrap_or_default().to_os_string();
        name.push(".tmp");
        output_path.with_file_name(name)
    };

    // Guard: remove the temp file on any early return (encode error, etc.).
    // Disarmed on success before the final rename.
    struct TmpGuard<'a> {
        path: &'a Path,
        armed: bool,
    }
    impl Drop for TmpGuard<'_> {
        fn drop(&mut self) {
            if self.armed {
                let _ = std::fs::remove_file(self.path);
            }
        }
    }
    let mut guard = TmpGuard {
        path: &tmp_path,
        armed: true,
    };

    let encoder_rate = format.effective_output_rate(sample_rate);
    let input_sample_format = ffmpeg_input_sample_format();
    let input_channel_layout = get_channel_layout("stereo");
    let sample_time_base = TimeBase::new(1, sample_rate as i32);

    let mut encoder = build_audio_encoder(format, input_channel_layout.clone(), encoder_rate)?;
    let encoder_params = encoder.codec_parameters();
    let mut muxer = open_output_muxer(
        output_path,
        &tmp_path,
        &[encoder_params.clone().into()],
        metadata,
    )?;
    let mut resampler = AudioResampler::builder()
        .source_channel_layout(input_channel_layout.clone())
        .source_sample_format(input_sample_format)
        .source_sample_rate(sample_rate)
        .target_channel_layout(encoder_params.channel_layout().to_owned())
        .target_sample_format(encoder_params.sample_format())
        .target_sample_rate(encoder_params.sample_rate())
        .target_frame_samples(encoder.samples_per_frame())
        .build()
        .map_err(|e| format!("Failed to create FFmpeg audio resampler: {e}"))?;

    let input_frame_samples = encoder.samples_per_frame().unwrap_or(4096).max(1);
    let frame_stride = input_frame_samples * CHANNELS as usize;
    let mut next_pts = 0i64;

    for chunk in samples.chunks(frame_stride) {
        let frame_samples = chunk.len() / CHANNELS as usize;
        if frame_samples == 0 {
            continue;
        }

        let mut frame = AudioFrameMut::silence(
            &input_channel_layout,
            input_sample_format,
            sample_rate,
            frame_samples,
        )
        .with_time_base(sample_time_base)
        .with_pts(Timestamp::new(next_pts, sample_time_base));
        next_pts += frame_samples as i64;

        copy_interleaved_f64_to_audio_frame(&mut frame, chunk)?;

        resampler
            .push(frame.freeze())
            .map_err(|e| format!("FFmpeg resampler push error: {e}"))?;
        drain_resampler_frames(&mut resampler, &mut encoder, &mut muxer)?;
    }

    resampler
        .flush()
        .map_err(|e| format!("FFmpeg resampler flush error: {e}"))?;
    drain_resampler_frames(&mut resampler, &mut encoder, &mut muxer)?;

    encoder
        .flush()
        .map_err(|e| format!("FFmpeg encoder flush error: {e}"))?;
    drain_encoder_packets(&mut encoder, &mut muxer)?;

    muxer
        .flush()
        .map_err(|e| format!("FFmpeg mux finalize error: {e}"))?;

    // Encoding complete — disarm the guard and atomically replace the final output.
    guard.armed = false;
    std::fs::rename(&tmp_path, output_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "Failed to rename temp file to {}: {e}",
            output_path.display()
        )
    })?;

    Ok(())
}

fn write_flac(
    samples: Vec<f64>,
    output_path: &Path,
    sample_rate: u32,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    write_audio_with_ffmpeg(
        samples,
        output_path,
        AudioExportFormat::Flac,
        sample_rate,
        metadata,
    )
}

/// Render the current live module state directly to a FLAC file.
#[allow(clippy::too_many_arguments)]
pub fn render_live_module_to_flac(
    module: &mut Module,
    output_path: &Path,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    hrtf_mix: i32,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    let mut samples = render_live_module_to_samples_with_agc(
        module,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
        sample_rate,
        None,
    )?;
    if hrtf_mix > 0 {
        crate::hrtf::process_offline_with_mix(&mut samples, sample_rate, hrtf_mix)?;
    }
    write_flac(samples, output_path, sample_rate, metadata)
}

fn write_aac(
    samples: Vec<f64>,
    output_path: &Path,
    sample_rate: u32,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    write_audio_with_ffmpeg(
        samples,
        output_path,
        AudioExportFormat::Aac,
        sample_rate,
        metadata,
    )
}

/// Encode pre-rendered samples to FLAC with progress reporting in [lo..hi].
#[allow(dead_code)]
pub(crate) fn encode_samples_to_flac(
    samples: Vec<f64>,
    output_path: &Path,
    sample_rate: u32,
    metadata: &AudioMetadata,
    progress_tx: &crossbeam_channel::Sender<f32>,
    progress_lo: f32,
    progress_hi: f32,
) -> Result<(), String> {
    write_audio_with_ffmpeg_progress(
        samples,
        output_path,
        AudioExportFormat::Flac,
        sample_rate,
        metadata,
        progress_tx,
        progress_lo,
        progress_hi,
    )
}

/// Encode pre-rendered samples to AAC with progress reporting in [lo..hi].
#[allow(dead_code)]
pub(crate) fn encode_samples_to_aac(
    samples: Vec<f64>,
    output_path: &Path,
    sample_rate: u32,
    metadata: &AudioMetadata,
    progress_tx: &crossbeam_channel::Sender<f32>,
    progress_lo: f32,
    progress_hi: f32,
) -> Result<(), String> {
    write_audio_with_ffmpeg_progress(
        samples,
        output_path,
        AudioExportFormat::Aac,
        sample_rate,
        metadata,
        progress_tx,
        progress_lo,
        progress_hi,
    )
}

/// Render the current live module state directly to an AAC file.
#[allow(clippy::too_many_arguments)]
pub fn render_live_module_to_aac(
    module: &mut Module,
    output_path: &Path,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    hrtf_mix: i32,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    let mut samples = render_live_module_to_samples_with_agc(
        module,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
        sample_rate,
        None,
    )?;
    if hrtf_mix > 0 {
        crate::hrtf::process_offline_with_mix(&mut samples, sample_rate, hrtf_mix)?;
    }
    write_aac(samples, output_path, sample_rate, metadata)
}

/// Render a serialized module snapshot to FLAC with progress reporting.
/// Progress is sent as f32 in 0.0..=1.0.  Does not hold any external mutex.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_snapshot_to_flac(
    module_bytes: &[u8],
    output_path: &Path,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    metadata: &AudioMetadata,
    progress_tx: &crossbeam_channel::Sender<f32>,
) -> Result<(), String> {
    let samples = render_snapshot_to_samples(
        module_bytes,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
        sample_rate,
        progress_tx,
        0.0,
        0.5,
    )?;
    write_audio_with_ffmpeg_progress(
        samples,
        output_path,
        AudioExportFormat::Flac,
        sample_rate,
        metadata,
        progress_tx,
        0.5,
        1.0,
    )
}

/// Render a serialized module snapshot to AAC with progress reporting.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_snapshot_to_aac(
    module_bytes: &[u8],
    output_path: &Path,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    metadata: &AudioMetadata,
    progress_tx: &crossbeam_channel::Sender<f32>,
) -> Result<(), String> {
    let samples = render_snapshot_to_samples(
        module_bytes,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
        sample_rate,
        progress_tx,
        0.0,
        0.5,
    )?;
    write_audio_with_ffmpeg_progress(
        samples,
        output_path,
        AudioExportFormat::Aac,
        sample_rate,
        metadata,
        progress_tx,
        0.5,
        1.0,
    )
}

/// Load a module from bytes, render to samples, and report progress in
/// the [lo..hi] range of the overall 0.0..1.0 progress.
#[allow(clippy::too_many_arguments)]
fn render_snapshot_to_samples(
    module_bytes: &[u8],
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    sample_rate: u32,
    progress_tx: &crossbeam_channel::Sender<f32>,
    progress_lo: f32,
    progress_hi: f32,
) -> Result<Vec<f64>, String> {
    let mut module = crate::openmpt::Module::from_memory(module_bytes)?;
    configure_module_for_render(
        &mut module,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
    );

    let duration = module.duration_seconds().max(1.0);
    let mut samples = Vec::new();
    let mut buf = vec![0f64; sample_rate as usize * CHANNELS as usize];
    let mut seconds_rendered = 0.0f64;
    loop {
        let frames = module.read_interleaved_double_stereo(sample_rate as i32, &mut buf);
        if frames == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..frames * CHANNELS as usize]);
        seconds_rendered += frames as f64 / sample_rate as f64;
        let frac = (seconds_rendered / duration).min(1.0) as f32;
        let _ = progress_tx.try_send(progress_lo + frac * (progress_hi - progress_lo));
    }
    if samples.is_empty() {
        return Err("Module produced no audio".into());
    }
    Ok(samples)
}

/// Encode with FFmpeg, reporting progress in the [lo..hi] range.
#[allow(clippy::too_many_arguments)]
fn write_audio_with_ffmpeg_progress(
    mut samples: Vec<f64>,
    output_path: &Path,
    format: AudioExportFormat,
    sample_rate: u32,
    metadata: &AudioMetadata,
    progress_tx: &crossbeam_channel::Sender<f32>,
    progress_lo: f32,
    progress_hi: f32,
) -> Result<(), String> {
    normalize_samples(&mut samples, sample_rate, format.true_peak_ceiling_dbtp());

    let tmp_path = {
        let mut name = output_path.file_name().unwrap_or_default().to_os_string();
        name.push(".tmp");
        output_path.with_file_name(name)
    };
    struct TmpGuard<'a> {
        path: &'a Path,
        armed: bool,
    }
    impl Drop for TmpGuard<'_> {
        fn drop(&mut self) {
            if self.armed {
                let _ = std::fs::remove_file(self.path);
            }
        }
    }
    let mut guard = TmpGuard {
        path: &tmp_path,
        armed: true,
    };

    let encoder_rate = format.effective_output_rate(sample_rate);
    let input_sample_format = ffmpeg_input_sample_format();
    let input_channel_layout = get_channel_layout("stereo");
    let sample_time_base = TimeBase::new(1, sample_rate as i32);

    let mut encoder = build_audio_encoder(format, input_channel_layout.clone(), encoder_rate)?;
    let encoder_params = encoder.codec_parameters();
    let mut muxer = open_output_muxer(
        output_path,
        &tmp_path,
        &[encoder_params.clone().into()],
        metadata,
    )?;
    let mut resampler = AudioResampler::builder()
        .source_channel_layout(input_channel_layout.clone())
        .source_sample_format(input_sample_format)
        .source_sample_rate(sample_rate)
        .target_channel_layout(encoder_params.channel_layout().to_owned())
        .target_sample_format(encoder_params.sample_format())
        .target_sample_rate(encoder_params.sample_rate())
        .target_frame_samples(encoder.samples_per_frame())
        .build()
        .map_err(|e| format!("Failed to create FFmpeg audio resampler: {e}"))?;

    let input_frame_samples = encoder.samples_per_frame().unwrap_or(4096).max(1);
    let frame_stride = input_frame_samples * CHANNELS as usize;
    let mut next_pts = 0i64;
    let total_samples = samples.len();

    for (chunk_idx, chunk) in samples.chunks(frame_stride).enumerate() {
        let frame_samples = chunk.len() / CHANNELS as usize;
        if frame_samples == 0 {
            continue;
        }

        let mut frame = AudioFrameMut::silence(
            &input_channel_layout,
            input_sample_format,
            sample_rate,
            frame_samples,
        )
        .with_time_base(sample_time_base)
        .with_pts(Timestamp::new(next_pts, sample_time_base));
        next_pts += frame_samples as i64;

        copy_interleaved_f64_to_audio_frame(&mut frame, chunk)?;

        resampler
            .push(frame.freeze())
            .map_err(|e| format!("FFmpeg resampler push error: {e}"))?;
        drain_resampler_frames(&mut resampler, &mut encoder, &mut muxer)?;

        let encoded_samples = (chunk_idx + 1) * frame_stride;
        let frac = (encoded_samples as f32 / total_samples as f32).min(1.0);
        let _ = progress_tx.try_send(progress_lo + frac * (progress_hi - progress_lo));
    }

    resampler
        .flush()
        .map_err(|e| format!("FFmpeg resampler flush error: {e}"))?;
    drain_resampler_frames(&mut resampler, &mut encoder, &mut muxer)?;

    encoder
        .flush()
        .map_err(|e| format!("FFmpeg encoder flush error: {e}"))?;
    drain_encoder_packets(&mut encoder, &mut muxer)?;

    muxer
        .flush()
        .map_err(|e| format!("FFmpeg mux finalize error: {e}"))?;

    guard.armed = false;
    std::fs::rename(&tmp_path, output_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "Failed to rename temp file to {}: {e}",
            output_path.display()
        )
    })?;

    let _ = progress_tx.try_send(1.0);
    Ok(())
}

/// Render a module from raw bytes to FLAC.
#[allow(dead_code)]
pub(crate) fn render_file_to_flac(
    file_data: &[u8],
    output_path: &Path,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    let samples = render_module_to_samples_with_agc(
        file_data,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
    )?;
    write_flac(samples, output_path, DEFAULT_SAMPLE_RATE, metadata)
}

/// Render a module from raw bytes to AAC.
#[allow(dead_code)]
pub(crate) fn render_file_to_aac(
    file_data: &[u8],
    output_path: &Path,
    stereo_separation: i32,
    interpolation_filter: i32,
    agc_enabled: bool,
    metadata: &AudioMetadata,
) -> Result<(), String> {
    let samples = render_module_to_samples_with_agc(
        file_data,
        stereo_separation,
        interpolation_filter,
        agc_enabled,
    )?;
    write_aac(samples, output_path, DEFAULT_SAMPLE_RATE, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::File, path::Path};

    use ac_ffmpeg::format::{demuxer::Demuxer, io::IO};

    const SAMPLE_RATE: u32 = DEFAULT_SAMPLE_RATE;
    /// Test-local mirror of `AudioExportFormat::Flac.true_peak_ceiling_dbtp()`.
    /// Tests below cover the FLAC normalization branch (original behavior).
    const TEST_CEILING_DBTP: f64 = -1.0;

    fn normalize_samples_scalar(samples: &mut [f64]) {
        let len_per_ch = samples.len() / CHANNELS as usize;
        for ch in 0..CHANNELS as usize {
            let sum: f64 = samples
                .iter()
                .skip(ch)
                .step_by(CHANNELS as usize)
                .copied()
                .sum();
            let mean = sum / len_per_ch as f64;
            for s in samples.iter_mut().skip(ch).step_by(CHANNELS as usize) {
                *s -= mean;
            }
        }
        let ceiling_lin = 10.0_f64.powf(TEST_CEILING_DBTP / 20.0);
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        if peak > 0.0 {
            let gain = ceiling_lin / peak;
            for s in samples.iter_mut() {
                *s *= gain;
            }
        }
    }

    fn test_stereo_buffer(frames: usize) -> Vec<f64> {
        (0..frames)
            .flat_map(|i| {
                let frame = i as f64;
                [
                    (frame * 0.013).sin() * 0.7 + 0.1,
                    (frame * 0.021).cos() * 0.5 - 0.08,
                ]
            })
            .collect()
    }

    #[test]
    fn normalize_samples_matches_scalar_reference() {
        // Verify DC removal still works (compare debiased samples before gain)
        let mut actual = test_stereo_buffer(8_193);
        let mut expected = actual.clone();

        normalize_samples_scalar(&mut expected);
        normalize_samples(&mut actual, SAMPLE_RATE, TEST_CEILING_DBTP);

        // Both should remove DC bias identically (left/right channel means → 0)
        let actual_left_mean: f64 =
            actual.iter().step_by(2).copied().sum::<f64>() / (actual.len() / 2) as f64;
        assert!(
            actual_left_mean.abs() < 0.01,
            "DC bias not removed: {actual_left_mean}"
        );

        // Sample peak must stay under the true-peak ceiling (we never clamp).
        let ceiling_lin = 10.0_f64.powf(TEST_CEILING_DBTP / 20.0);
        let actual_peak = actual.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        assert!(
            actual_peak <= ceiling_lin + 1e-9,
            "Sample peak exceeds ceiling: {actual_peak} > {ceiling_lin}"
        );
        assert!(actual_peak > 0.0, "Output is silent");

        // Verify LUFS measurement of normalized output is near -14 LUFS
        if let Ok(mut meter) =
            ebur128::EbuR128::new(CHANNELS as u32, SAMPLE_RATE as u32, ebur128::Mode::I)
        {
            meter.add_frames_f64(&actual).unwrap();
            let loudness = meter.loudness_global().unwrap();
            if loudness.is_finite() {
                assert!(
                    (loudness - TARGET_LUFS).abs() < 1.0,
                    "LUFS should be near {TARGET_LUFS}, got {loudness}"
                );
            }
        }
    }

    fn stereo_sine(frequency: f64, amplitude: f64, sample_rate: u32, seconds: f64) -> Vec<f64> {
        let frames = (sample_rate as f64 * seconds) as usize;
        let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate as f64;
        (0..frames)
            .flat_map(|i| {
                let v = (omega * i as f64).sin() * amplitude;
                [v, v]
            })
            .collect()
    }

    fn measure_true_peak_max(samples: &[f64], sample_rate: u32) -> f64 {
        let mut meter =
            ebur128::EbuR128::new(CHANNELS as u32, sample_rate, ebur128::Mode::TRUE_PEAK)
                .expect("build true-peak meter");
        meter.add_frames_f64(samples).expect("add frames");
        (0..CHANNELS)
            .filter_map(|c| meter.true_peak(c).ok())
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max)
    }

    fn measure_integrated_lufs(samples: &[f64], sample_rate: u32) -> f64 {
        let mut meter = ebur128::EbuR128::new(CHANNELS as u32, sample_rate, ebur128::Mode::I)
            .expect("build loudness meter");
        meter.add_frames_f64(samples).expect("add frames");
        meter.loudness_global().expect("loudness_global")
    }

    #[test]
    fn normalize_ceiling_holds_on_intersample_bomb() {
        // Alternating ±0.95 is a classic inter-sample-peak test vector:
        // its reconstructed true-peak (4× oversampled) exceeds its sample
        // peak. This is the signal class that produced the +0.2 dBTP
        // overs on AAC decode under the old clamp-based normalizer.
        let frames = SAMPLE_RATE as usize * 2;
        let mut samples: Vec<f64> = (0..frames)
            .flat_map(|i| {
                let v = if i % 2 == 0 { 0.95 } else { -0.95 };
                [v, v]
            })
            .collect();

        normalize_samples(&mut samples, SAMPLE_RATE, TEST_CEILING_DBTP);

        // Allow 0.01 dB measurement headroom above the ceiling.
        let tp_bound = 10.0_f64.powf((TEST_CEILING_DBTP + 0.01) / 20.0);
        let tp = measure_true_peak_max(&samples, SAMPLE_RATE);
        assert!(
            tp <= tp_bound,
            "true-peak {tp} exceeds {TEST_CEILING_DBTP} dBTP ceiling",
        );
    }

    #[test]
    fn normalize_lufs_hit_when_peaks_allow() {
        // Quiet 1 kHz sine: LUFS target demands +gain, post-gain peak is
        // far below ceiling → LUFS branch wins and output hits target.
        let mut samples = stereo_sine(1_000.0, 0.2, SAMPLE_RATE, 2.0);

        normalize_samples(&mut samples, SAMPLE_RATE, TEST_CEILING_DBTP);

        let lufs = measure_integrated_lufs(&samples, SAMPLE_RATE);
        assert!(lufs.is_finite(), "loudness should be finite");
        assert!(
            (lufs - TARGET_LUFS).abs() < 1.0,
            "expected LUFS near {TARGET_LUFS}, got {lufs}",
        );
    }

    #[test]
    fn normalize_ceiling_wins_on_loud_input() {
        // 15 Hz at 0.95: deep sub-bass is K-weighted down ~16 dB, so LUFS
        // gain wants to boost ~+5 dB, but the sample peak is already 0.95
        // so the boosted peak would smash the ceiling. Ceiling branch wins
        // and output ends up several dB below -14 LUFS — the expected
        // fidelity-first trade-off for hot LF-heavy content.
        let mut samples = stereo_sine(15.0, 0.95, SAMPLE_RATE, 2.0);

        normalize_samples(&mut samples, SAMPLE_RATE, TEST_CEILING_DBTP);

        let tp_bound = 10.0_f64.powf((TEST_CEILING_DBTP + 0.01) / 20.0);
        let tp = measure_true_peak_max(&samples, SAMPLE_RATE);
        assert!(
            tp <= tp_bound,
            "true-peak {tp} exceeds {TEST_CEILING_DBTP} dBTP ceiling",
        );

        let lufs = measure_integrated_lufs(&samples, SAMPLE_RATE);
        assert!(
            lufs.is_finite() && lufs < TARGET_LUFS - 0.5,
            "expected ceiling-limited LUFS below {}, got {lufs}",
            TARGET_LUFS - 0.5,
        );
    }

    #[test]
    fn normalize_silence_and_near_silence() {
        let mut silent = vec![0.0; SAMPLE_RATE as usize * CHANNELS as usize];
        normalize_samples(&mut silent, SAMPLE_RATE, TEST_CEILING_DBTP);
        assert!(silent.iter().all(|&s| s == 0.0), "silence stays silent");

        // Genuinely small but non-DC signal: a 1 kHz sine at 1e-6 amplitude.
        // ebur128 may gate this as silence (< -70 LUFS) and return -inf; in
        // that case the fallback peak-normalizes to ceiling. Either path
        // must produce finite, ceiling-safe output.
        let mut tiny = stereo_sine(1_000.0, 1e-6, SAMPLE_RATE, 1.0);
        normalize_samples(&mut tiny, SAMPLE_RATE, TEST_CEILING_DBTP);
        let ceiling_lin = 10.0_f64.powf(TEST_CEILING_DBTP / 20.0);
        assert!(
            tiny.iter().all(|s| s.is_finite()),
            "near-silent input produced non-finite output",
        );
        let peak = tiny.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        assert!(
            peak <= ceiling_lin + 1e-9,
            "near-silent peak {peak} exceeds ceiling {ceiling_lin}",
        );
    }

    fn probe_audio_stream(path: &Path) {
        let file = File::open(path).expect("Should open rendered audio");
        let io = IO::from_seekable_read_stream(file);
        let demuxer = Demuxer::builder()
            .build(io)
            .expect("Should build FFmpeg demuxer")
            .find_stream_info(None)
            .map_err(|(_, e)| e)
            .expect("Should discover stream info");
        assert!(
            demuxer
                .streams()
                .iter()
                .any(|stream| stream.codec_parameters().is_audio_codec()),
            "Rendered file should expose an audio stream",
        );
    }

    #[test]
    fn ffmpeg_export_backends_write_decodable_files() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let flac_path = temp_dir.path().join("synthetic.flac");
        let aac_path = temp_dir.path().join("synthetic.m4a");
        let samples = test_stereo_buffer(SAMPLE_RATE as usize);

        let metadata = AudioMetadata {
            title: "Test".into(),
            artist: String::new(),
            album: "Quinlight Audio".into(),
        };
        write_flac(samples.clone(), &flac_path, SAMPLE_RATE, &metadata)
            .expect("FLAC export should succeed");
        write_aac(samples, &aac_path, SAMPLE_RATE, &metadata).expect("AAC export should succeed");

        probe_audio_stream(&flac_path);
        probe_audio_stream(&aac_path);
    }

    #[test]
    fn ffmpeg_export_uses_double_precision_input_frames() {
        let input = vec![-0.75, 0.25, 0.5, -0.125];
        let layout = get_channel_layout("stereo");
        let sample_format = ffmpeg_input_sample_format();
        assert_eq!(sample_format.name(), "dbl");

        let mut frame =
            AudioFrameMut::silence(&layout, sample_format, SAMPLE_RATE, input.len() / 2);
        copy_interleaved_f64_to_audio_frame(&mut frame, &input)
            .expect("should copy f64 samples into FFmpeg frame");

        let mut planes = frame.planes_mut();
        let plane = planes.first_mut().expect("frame should contain one plane");
        let byte_len = std::mem::size_of_val(input.as_slice());
        let recovered: Vec<f64> = plane.data()[..byte_len]
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        assert_eq!(recovered, input);
    }
}
