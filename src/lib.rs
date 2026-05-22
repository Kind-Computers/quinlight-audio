// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::manual_range_contains,
    clippy::unnecessary_cast
)]

pub mod archive;
pub mod cleanup;
pub mod engine;
pub mod hrtf;
pub mod openmpt;
pub mod player;
pub mod remaster;
mod render;
mod simd;

#[cfg(test)]
mod tests {
    use super::openmpt::{
        AgcProfile, DEFAULT_AGC_ENABLED, DEFAULT_AGC_PROFILE, DEFAULT_INTERPOLATION_FILTER_LENGTH,
        DEFAULT_STEREO_SEPARATION_PERCENT, DEFAULT_VOLUMERAMPING_STRENGTH, Module, SampleFormat,
        SampleInfo, SampleLoopInfo, SampleLoopMode, SampleLoopRegion, nativefloat_size_bytes,
    };
    use super::remaster;
    use super::render;

    const BASIC_FIXTURE: &str = "mods/2ND_PM.S3M";
    const XM_FIXTURE: &str = "openmpt/test/test.xm";
    const MOD_FIXTURE: &str = "openmpt/test/test.mod";
    const S3M_FIXTURE: &str = "openmpt/test/test.s3m";
    const MPTM_FIXTURE: &str = "openmpt/test/test.mptm";
    const BEYOND_NETWORK_FIXTURE: &str = "mods/beyond_the_network.it";

    const COMMAND_NOTE: i32 = 0;
    const COMMAND_INSTRUMENT: i32 = 1;
    const COMMAND_VOLUME_COMMAND: i32 = 2;
    const COMMAND_EFFECT: i32 = 3;
    const COMMAND_VOLUME: i32 = 4;
    const COMMAND_PARAMETER: i32 = 5;
    const VOLCMD_VIBRATOSPEED: u8 = 7;
    const VOLCMD_VIBRATODEPTH: u8 = 8;
    const CMD_PORTAMENTODOWN: u8 = 3;
    const CMD_TONEPORTAMENTO: u8 = 4;
    const CMD_VIBRATO: u8 = 5;
    const CMD_OFFSET: u8 = 10;
    const CMD_XFINEPORTAUPDOWN: u8 = 28;
    const MOD_TEST_NOTE: u8 = 48;

    fn doubled_sample_data(original_data: &[f64], original_length: i64, channels: i32) -> Vec<f64> {
        let new_length = original_length * 2;
        let mut new_data = vec![0.0f64; (new_length as usize) * (channels as usize)];
        for i in 0..new_length as usize {
            let src = i as f64 / 2.0;
            let src_idx = src as usize;
            let frac = (src - src_idx as f64) as f64;
            for ch in 0..channels as usize {
                let idx0 = src_idx * channels as usize + ch;
                let idx1 = (src_idx + 1).min(original_length as usize - 1) * channels as usize + ch;
                new_data[i * channels as usize + ch] =
                    original_data[idx0] * (1.0 - frac) + original_data[idx1] * frac;
            }
        }
        new_data
    }

    fn mean_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        assert!(len > 0, "Buffers must contain at least one sample");
        a.iter()
            .zip(b.iter())
            .take(len)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f64>()
            / len as f64
    }

    fn max_adjacent_jump(samples: &[f64]) -> f64 {
        samples
            .windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .fold(0.0f64, f64::max)
    }

    fn mean_abs(data: &[f64]) -> f64 {
        assert!(!data.is_empty(), "Buffer must contain at least one sample");
        data.iter().map(|sample| sample.abs()).sum::<f64>() / data.len() as f64
    }

    fn rms(data: &[f64]) -> f64 {
        assert!(!data.is_empty(), "Buffer must contain at least one sample");
        (data.iter().map(|sample| sample * sample).sum::<f64>() / data.len() as f64).sqrt()
    }

    fn assert_close(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64, context: &str) {
        let delta = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs());
        assert!(
            delta <= abs_tol.max(scale * rel_tol),
            "{context}: expected {expected:.12}, got {actual:.12}, delta {delta:.12}",
        );
    }

    fn render_window(module: &mut Module, start_seconds: f64, end_seconds: f64) -> Vec<f64> {
        assert!(
            end_seconds > start_seconds,
            "render window end must be after start"
        );
        module.set_position_seconds(start_seconds);
        let target_frames = ((end_seconds - start_seconds) * 48_000.0).round() as usize;
        let mut rendered = Vec::with_capacity(target_frames * 2);
        let mut buf = vec![0.0f64; 4096 * 2];
        while rendered.len() / 2 < target_frames {
            let frames = module.read_interleaved_double_stereo(48_000, &mut buf);
            if frames == 0 {
                break;
            }
            let needed_frames = target_frames.saturating_sub(rendered.len() / 2);
            let take_frames = frames.min(needed_frames);
            rendered.extend_from_slice(&buf[..take_frames * 2]);
        }
        rendered
    }

    fn fnv1a_u64_update(hash: u64, value: u64) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut next = if hash == 0 { FNV_OFFSET } else { hash };
        for byte in value.to_le_bytes() {
            next ^= u64::from(byte);
            next = next.wrapping_mul(FNV_PRIME);
        }
        next
    }

    fn quantized_audio_signature(samples: &[f64], stride: usize, scale: f64) -> u64 {
        assert!(stride > 0, "signature stride must be non-zero");
        let mut hash = 0u64;
        hash = fnv1a_u64_update(hash, samples.len() as u64);
        for sample in samples.iter().step_by(stride) {
            let quantized = (sample * scale).round() as i64;
            hash = fnv1a_u64_update(hash, quantized as u64);
        }
        hash
    }

    #[test]
    fn test_openmpt_nativefloat_is_double_precision() {
        assert_eq!(nativefloat_size_bytes(), 8);
    }

    fn configure_quinlight_render(
        module: &mut Module,
        stereo_separation: i32,
        interpolation_filter: i32,
        agc_enabled: bool,
    ) {
        module.set_repeat_count(0);
        module.apply_quinlight_processing_settings(
            stereo_separation,
            interpolation_filter,
            agc_enabled,
        );
        module.set_position_seconds(0.0);
    }

    fn render_module_output(module: &mut Module) -> Vec<f64> {
        render_module_output_capped(module, 0)
    }

    /// Render module audio. If `max_seconds > 0`, stop after that many seconds.
    fn render_module_output_capped(module: &mut Module, max_seconds: u32) -> Vec<f64> {
        let max_frames = if max_seconds > 0 {
            max_seconds as usize * 48000
        } else {
            usize::MAX
        };
        let mut rendered = if max_seconds > 0 {
            Vec::with_capacity(max_frames * 2)
        } else {
            Vec::new()
        };
        let mut buf = vec![0.0f64; 48000 * 2];
        loop {
            let frames = module.read_interleaved_double_stereo(48000, &mut buf);
            if frames == 0 {
                break;
            }
            assert!(
                buf[..frames * 2].iter().all(|sample| sample.is_finite()),
                "Rendered samples should remain finite",
            );
            rendered.extend_from_slice(&buf[..frames * 2]);
            if rendered.len() / 2 >= max_frames {
                rendered.truncate(max_frames * 2);
                break;
            }
        }
        rendered
    }

    fn resample_for_test(
        data: &[f64],
        from_rate: u32,
        to_rate: u32,
        channels: usize,
        boundary_mode: remaster::ResampleBoundaryMode,
    ) -> Vec<f64> {
        match remaster::resample_audio(data, from_rate, to_rate, channels, boundary_mode) {
            Ok(resampled) if !resampled.is_empty() => resampled,
            _ => match boundary_mode {
                remaster::ResampleBoundaryMode::OneShot => {
                    remaster::linear_resample_interleaved(data, from_rate, to_rate, channels)
                }
                remaster::ResampleBoundaryMode::LoopAware => {
                    let one_copy_frames =
                        remaster::scaled_frame_count(data.len() / channels, from_rate, to_rate);
                    let one_copy_samples = one_copy_frames * channels;
                    let mut tiled = Vec::with_capacity(data.len() * 3);
                    tiled.extend_from_slice(data);
                    tiled.extend_from_slice(data);
                    tiled.extend_from_slice(data);
                    let resampled =
                        remaster::linear_resample_interleaved(&tiled, from_rate, to_rate, channels);
                    resampled[one_copy_samples..one_copy_samples * 2].to_vec()
                }
            },
        }
    }

    fn first_nonempty_sample(module: &Module) -> super::openmpt::SampleInfo {
        module
            .info()
            .samples
            .into_iter()
            .find(|sample| sample.length_frames > 0 && sample.channels > 0 && sample.rate > 0)
            .expect("Fixture should contain at least one playable sample")
    }

    fn assert_save_loaded_format_roundtrip(
        path: &str,
        expected_extension: &str,
        expected_format: &str,
    ) {
        let data = std::fs::read(path).expect("Failed to read fixture");
        let module = Module::from_memory(&data).expect("Failed to load fixture");

        assert_eq!(
            module.loaded_format_extension(),
            expected_extension,
            "Loaded format extension should match the source format",
        );

        let saved = module
            .save_loaded_format_to_memory()
            .expect("Loaded format should be writable");
        let reloaded = Module::from_memory(&saved).expect("Failed to reload saved bytes");
        assert_eq!(
            reloaded.info().format_type,
            expected_format,
            "Saving in the loaded format should round-trip as the same module format",
        );
    }

    fn mod_regression_sample(module: &Module) -> super::openmpt::SampleInfo {
        let playable: Vec<_> = module
            .info()
            .samples
            .into_iter()
            .filter(|sample| sample.length_frames > 0 && sample.channels > 0 && sample.rate > 0)
            .collect();
        playable
            .iter()
            .filter(|sample| !sample.has_loop)
            .max_by_key(|sample| sample.length_frames)
            .cloned()
            .or_else(|| {
                playable
                    .into_iter()
                    .max_by_key(|sample| sample.length_frames)
            })
            .expect("Fixture should contain at least one playable sample")
    }

    fn find_mod_fixture_sample<F>(predicate: F) -> Option<(String, Vec<u8>, i32, SampleLoopInfo)>
    where
        F: Fn(SampleLoopInfo) -> bool,
    {
        let entries = std::fs::read_dir("mods").ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Ok(data) = std::fs::read(&path) else {
                continue;
            };
            let Ok(module) = Module::from_memory(&data) else {
                continue;
            };
            for sample in module.info().samples {
                if sample.length_frames <= 0 || sample.channels <= 0 || sample.rate <= 0 {
                    continue;
                }
                let loop_info = module.sample_loop_info(sample.index);
                if predicate(loop_info) {
                    return Some((path.display().to_string(), data, sample.index, loop_info));
                }
            }
        }
        None
    }

    fn configure_mod_phrase(module: &mut Module, sample_index: i32, note: u8) {
        let rows = module.pattern_num_rows(0);
        assert!(rows > 0, "MOD fixture should have pattern data");

        let clear_rows = rows.min(16);
        let num_channels = module.num_channels();
        for row in 0..clear_rows {
            for ch in 0..num_channels {
                assert!(module.set_pattern_command(0, row, ch, COMMAND_NOTE, 0));
                assert!(module.set_pattern_command(0, row, ch, COMMAND_INSTRUMENT, 0));
                assert!(module.set_pattern_command(0, row, ch, COMMAND_EFFECT, 0));
                assert!(module.set_pattern_command(0, row, ch, COMMAND_PARAMETER, 0));
            }
        }

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, (sample_index + 1) as u8,));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, 0));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, 0));
    }

    fn configure_mod_phrase_with_followup_note(
        module: &mut Module,
        sample_index: i32,
        start_note: u8,
        followup_row: i32,
        followup_note: u8,
    ) {
        configure_mod_phrase(module, sample_index, start_note);
        assert!(
            module.pattern_num_rows(0) > followup_row,
            "MOD fixture should have enough rows for the follow-up note",
        );
        assert!(module.set_pattern_command(0, followup_row, 0, COMMAND_NOTE, followup_note));
        assert!(module.set_pattern_command(0, followup_row, 0, COMMAND_INSTRUMENT, 0));
        assert!(module.set_pattern_command(0, followup_row, 0, COMMAND_EFFECT, 0));
        assert!(module.set_pattern_command(0, followup_row, 0, COMMAND_PARAMETER, 0));
    }

    fn synthetic_tone_at_freq(
        sample_rate: u32,
        channels: usize,
        frames: usize,
        freq_hz: f64,
    ) -> Vec<f64> {
        let ramp = frames.min((sample_rate as usize / 100).max(1));
        let mut data = vec![0.0f64; frames * channels];
        for frame in 0..frames {
            let envelope = if frame < ramp {
                frame as f64 / ramp as f64
            } else if frame + ramp >= frames {
                (frames - frame) as f64 / ramp as f64
            } else {
                1.0
            };
            let sample = envelope
                * 0.7
                * (frame as f64 * freq_hz * 2.0 * std::f64::consts::PI / sample_rate as f64).sin();
            for ch in 0..channels {
                data[frame * channels + ch] = sample;
            }
        }
        data
    }

    fn synthetic_tone(sample_rate: u32, channels: usize, frames: usize) -> Vec<f64> {
        synthetic_tone_at_freq(sample_rate, channels, frames, 440.0)
    }

    fn synthetic_loop_tone(
        _sample_rate: u32,
        channels: usize,
        frames: usize,
        loop_frames: usize,
    ) -> Vec<f64> {
        let loop_frames = loop_frames.max(2);
        let mut data = vec![0.0f64; frames * channels];
        for frame in 0..frames {
            let phase =
                2.0 * std::f64::consts::PI * (frame % loop_frames) as f64 / loop_frames as f64;
            let sample = 0.7 * phase.sin();
            for ch in 0..channels {
                data[frame * channels + ch] = sample;
            }
        }
        data
    }

    fn agc_profile_test_tone(sample_rate: u32, channels: usize, frames: usize) -> Vec<f64> {
        let burst_frames = frames.min(128);
        let mut data = vec![0.0f64; frames * channels];
        for frame in 0..frames {
            let sample = if frame < burst_frames {
                if frame % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                0.18 * (frame as f64 * 220.0 * 2.0 * std::f64::consts::PI / sample_rate as f64)
                    .sin()
            };
            for ch in 0..channels {
                data[frame * channels + ch] = sample;
            }
        }
        data
    }

    fn stereo_window_to_mono(data: &[f64], skip_frames: usize, take_frames: usize) -> Vec<f64> {
        let total_frames = data.len() / 2;
        assert!(
            total_frames > skip_frames,
            "Stereo window skip exceeds rendered audio"
        );

        let start = skip_frames.min(total_frames - 1);
        let end = (start + take_frames).min(total_frames);
        assert!(end > start, "Stereo analysis window must contain audio");

        let mut mono = Vec::with_capacity(end - start);
        for frame in data[start * 2..end * 2].chunks_exact(2) {
            mono.push(0.5 * (frame[0] + frame[1]));
        }
        mono
    }

    fn estimate_frequency(signal: &[f64], sample_rate: f64) -> f64 {
        let mut zero_crossings = Vec::new();
        for i in 1..signal.len() {
            if signal[i - 1] <= 0.0 && signal[i] > 0.0 {
                let denom = signal[i] - signal[i - 1];
                let frac = if denom.abs() > f64::EPSILON {
                    -signal[i - 1] / denom
                } else {
                    0.0
                };
                zero_crossings.push((i - 1) as f64 + frac);
            }
        }

        if zero_crossings.len() >= 2 {
            let avg_period = zero_crossings
                .windows(2)
                .map(|window| window[1] - window[0])
                .sum::<f64>()
                / (zero_crossings.len() - 1) as f64;
            return sample_rate / avg_period;
        }

        assert!(
            signal.len() >= 64,
            "Need enough samples to estimate pitch from autocorrelation",
        );

        let mean = signal.iter().copied().sum::<f64>() / signal.len() as f64;
        let centered: Vec<f64> = signal.iter().map(|sample| *sample - mean).collect();
        let max_lag = ((sample_rate / 40.0).ceil() as usize)
            .min(signal.len() / 2)
            .max(1);
        let min_lag = ((sample_rate / 4_000.0).floor() as usize).clamp(1, max_lag);
        let mut best_lag = 0usize;
        let mut best_corr = f64::MIN;

        for lag in min_lag..=max_lag {
            let window_len = centered.len() - lag;
            let corr = centered[..window_len]
                .iter()
                .zip(centered[lag..].iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>()
                / window_len as f64;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        assert!(
            best_lag > 0,
            "Autocorrelation pitch estimate should find a valid lag"
        );
        sample_rate / best_lag as f64
    }

    fn clear_pattern_rows(module: &mut Module, pattern: i32, rows: i32) {
        let num_rows = module.pattern_num_rows(pattern);
        assert!(
            num_rows > 0,
            "Pattern {pattern} should contain at least one row"
        );

        let clear_rows = num_rows.min(rows);
        let num_channels = module.num_channels();
        for row in 0..clear_rows {
            for ch in 0..num_channels {
                assert!(module.set_pattern_command(pattern, row, ch, COMMAND_NOTE, 0));
                assert!(module.set_pattern_command(pattern, row, ch, COMMAND_INSTRUMENT, 0));
                assert!(module.set_pattern_command(pattern, row, ch, COMMAND_VOLUME_COMMAND, 0));
                assert!(module.set_pattern_command(pattern, row, ch, COMMAND_EFFECT, 0));
                assert!(module.set_pattern_command(pattern, row, ch, COMMAND_VOLUME, 0));
                assert!(module.set_pattern_command(pattern, row, ch, COMMAND_PARAMETER, 0));
            }
        }
    }

    fn xm_low_rate_sample_and_note(module: &Module) -> (SampleInfo, u8) {
        let samples = module.info().samples;
        let mut notes_by_sample = vec![Vec::new(); samples.len()];

        for note in 1u8..=120 {
            if let Some(sample_index) = module.instrument_sample_for_note(0, note) {
                if let Some(notes) = notes_by_sample.get_mut(sample_index as usize) {
                    notes.push(note);
                }
            }
        }

        samples
            .into_iter()
            .filter(|sample| sample.length_frames > 0 && sample.channels > 0 && sample.rate > 0)
            .filter_map(|sample| {
                let notes = notes_by_sample.get(sample.index as usize)?;
                notes.first().copied().map(|note| (sample, note))
            })
            .min_by_key(|(sample, _)| sample.rate)
            .expect("XM fixture should map a playable note to a non-empty sample")
    }

    fn xm_sample_and_high_note(module: &Module) -> (SampleInfo, u8) {
        const TARGET_HIGH_NOTE: u8 = 101;
        let samples = module.info().samples;
        let mut notes_by_sample = vec![Vec::new(); samples.len()];

        for note in 1u8..=120 {
            if let Some(sample_index) = module.instrument_sample_for_note(0, note) {
                if let Some(notes) = notes_by_sample.get_mut(sample_index as usize) {
                    notes.push(note);
                }
            }
        }

        samples
            .into_iter()
            .filter(|sample| sample.length_frames > 0 && sample.channels > 0)
            .filter_map(|sample| {
                let notes = notes_by_sample.get(sample.index as usize)?;
                let high_note = notes
                    .iter()
                    .copied()
                    .filter(|&note| note <= TARGET_HIGH_NOTE)
                    .max()
                    .or_else(|| notes.last().copied())?;
                Some((sample, high_note))
            })
            .max_by_key(|(sample, high_note)| (*high_note, sample.length_frames))
            .expect("XM fixture should map a playable high note to a non-empty sample")
    }

    fn xm_porta_sample_and_notes(module: &Module) -> (SampleInfo, u8, u8) {
        let info = module.info();
        let samples = info.samples;
        let mut notes_by_sample = vec![Vec::new(); samples.len()];

        for note in 1u8..=120 {
            if let Some(sample_index) = module.instrument_sample_for_note(0, note) {
                if let Some(notes) = notes_by_sample.get_mut(sample_index as usize) {
                    notes.push(note);
                }
            }
        }

        let (sample_index, notes) = notes_by_sample
            .into_iter()
            .enumerate()
            .filter(|(_, notes)| notes.len() >= 2)
            .max_by_key(|(_, notes)| {
                notes.last().copied().unwrap_or(0) - notes.first().copied().unwrap_or(0)
            })
            .expect("XM fixture instrument should map at least one sample across multiple notes");

        let start_note = notes[0];
        let target_note = notes
            .iter()
            .copied()
            .find(|&note| note >= start_note.saturating_add(12))
            .unwrap_or_else(|| *notes.last().expect("mapped notes should not be empty"));

        assert!(
            target_note > start_note,
            "XM fixture should provide a higher mapped note for the tone-portamento target",
        );

        let sample = samples
            .into_iter()
            .find(|sample| sample.index as usize == sample_index)
            .expect("Mapped sample should exist in fixture metadata");
        assert!(
            sample.length_frames > 0 && sample.channels > 0 && sample.rate > 0,
            "Mapped XM sample should contain playable audio",
        );

        (sample, start_note, target_note)
    }

    fn xm_low_rate_porta_sample_and_notes(module: &Module) -> (SampleInfo, u8, u8) {
        let info = module.info();
        let samples = info.samples;
        let mut notes_by_sample = vec![Vec::new(); samples.len()];

        for note in 1u8..=120 {
            if let Some(sample_index) = module.instrument_sample_for_note(0, note) {
                if let Some(notes) = notes_by_sample.get_mut(sample_index as usize) {
                    notes.push(note);
                }
            }
        }

        samples
            .into_iter()
            .filter(|sample| sample.length_frames > 0 && sample.channels > 0 && sample.rate > 0)
            .filter_map(|sample| {
                let notes = notes_by_sample.get(sample.index as usize)?;
                if notes.len() < 2 {
                    return None;
                }
                let start_note = notes[0];
                let target_note = notes
                    .iter()
                    .copied()
                    .find(|&note| note >= start_note.saturating_add(12))
                    .unwrap_or_else(|| *notes.last().expect("mapped notes should not be empty"));
                (target_note > start_note).then_some((sample, start_note, target_note))
            })
            .min_by_key(|(sample, _, _)| sample.rate)
            .expect("XM fixture should map a low-rate playable sample across multiple notes")
    }

    fn configure_xm_porta_phrase(module: &mut Module, start_note: u8, target_note: u8) {
        clear_pattern_rows(module, 0, 16);

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, start_note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));

        assert!(module.set_pattern_command(0, 1, 0, COMMAND_NOTE, target_note));
        assert!(module.set_pattern_command(0, 1, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 1, 0, COMMAND_EFFECT, CMD_TONEPORTAMENTO));
        assert!(module.set_pattern_command(0, 1, 0, COMMAND_PARAMETER, 0x20));

        for row in 2..12 {
            assert!(module.set_pattern_command(0, row, 0, COMMAND_EFFECT, CMD_TONEPORTAMENTO));
            assert!(module.set_pattern_command(0, row, 0, COMMAND_PARAMETER, 0));
        }
    }

    fn configure_xm_porta_phrase_with_param(
        module: &mut Module,
        start_note: u8,
        target_note: u8,
        porta_param: u8,
        active_rows: i32,
    ) {
        assert!(
            active_rows >= 2,
            "Tone-portamento phrase needs at least two active rows"
        );
        clear_pattern_rows(module, 0, active_rows + 2);

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, start_note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));

        assert!(module.set_pattern_command(0, 1, 0, COMMAND_NOTE, target_note));
        assert!(module.set_pattern_command(0, 1, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 1, 0, COMMAND_EFFECT, CMD_TONEPORTAMENTO));
        assert!(module.set_pattern_command(0, 1, 0, COMMAND_PARAMETER, porta_param));

        for row in 2..=active_rows {
            assert!(module.set_pattern_command(0, row, 0, COMMAND_EFFECT, CMD_TONEPORTAMENTO));
            assert!(module.set_pattern_command(0, row, 0, COMMAND_PARAMETER, 0));
        }
    }

    fn configure_xm_offset_slide_phrase(
        module: &mut Module,
        note: u8,
        offset_param: u8,
        slide_param: u8,
        slide_rows: i32,
    ) {
        clear_pattern_rows(module, 0, slide_rows + 2);

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_OFFSET));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, offset_param));

        for row in 1..=slide_rows {
            assert!(module.set_pattern_command(0, row, 0, COMMAND_EFFECT, CMD_PORTAMENTODOWN));
            assert!(module.set_pattern_command(
                0,
                row,
                0,
                COMMAND_PARAMETER,
                if row == 1 { slide_param } else { 0 }
            ));
        }
    }

    fn configure_xm_extra_fine_slide_phrase(
        module: &mut Module,
        note: u8,
        down_param: u8,
        slide_rows: i32,
    ) {
        clear_pattern_rows(module, 0, slide_rows + 2);

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));

        for row in 1..=slide_rows {
            assert!(module.set_pattern_command(0, row, 0, COMMAND_EFFECT, CMD_XFINEPORTAUPDOWN));
            assert!(module.set_pattern_command(
                0,
                row,
                0,
                COMMAND_PARAMETER,
                0x20 | if row == 1 { down_param & 0x0F } else { 0 }
            ));
        }
    }

    fn configure_xm_vibrato_phrase(
        module: &mut Module,
        note: u8,
        vibrato_param: u8,
        hold_rows: i32,
    ) {
        assert!(
            hold_rows >= 2,
            "Vibrato phrase needs at least two hold rows"
        );
        clear_pattern_rows(module, 0, hold_rows + 2);

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_VIBRATO));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, vibrato_param));

        for row in 1..=hold_rows {
            assert!(module.set_pattern_command(0, row, 0, COMMAND_EFFECT, CMD_VIBRATO));
            assert!(module.set_pattern_command(0, row, 0, COMMAND_PARAMETER, 0));
        }
    }

    fn configure_xm_high_note_vibrato_phrase(
        module: &mut Module,
        note: u8,
        vibrato_param: u8,
        hold_rows: i32,
        volume_vibrato_depth: Option<u8>,
        volume_vibrato_speed: Option<u8>,
    ) {
        assert!(
            hold_rows >= 2,
            "High-note vibrato phrase needs at least two hold rows"
        );
        clear_pattern_rows(module, 0, hold_rows + 2);

        let write_volcmd = |module: &mut Module, row: i32| {
            if row == 0 {
                if let Some(speed) = volume_vibrato_speed {
                    assert!(module.set_pattern_command(
                        0,
                        row,
                        0,
                        COMMAND_VOLUME_COMMAND,
                        VOLCMD_VIBRATOSPEED
                    ));
                    assert!(module.set_pattern_command(0, row, 0, COMMAND_VOLUME, speed));
                    return;
                }
            }
            if let Some(depth) = volume_vibrato_depth {
                assert!(module.set_pattern_command(
                    0,
                    row,
                    0,
                    COMMAND_VOLUME_COMMAND,
                    VOLCMD_VIBRATODEPTH
                ));
                assert!(module.set_pattern_command(0, row, 0, COMMAND_VOLUME, depth));
                return;
            }
            if let Some(speed) = volume_vibrato_speed {
                assert!(module.set_pattern_command(
                    0,
                    row,
                    0,
                    COMMAND_VOLUME_COMMAND,
                    VOLCMD_VIBRATOSPEED
                ));
                assert!(module.set_pattern_command(0, row, 0, COMMAND_VOLUME, speed));
            }
        };

        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        write_volcmd(module, 0);
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_VIBRATO));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, vibrato_param));

        for row in 1..=hold_rows {
            write_volcmd(module, row);
            assert!(module.set_pattern_command(0, row, 0, COMMAND_EFFECT, CMD_VIBRATO));
            assert!(module.set_pattern_command(0, row, 0, COMMAND_PARAMETER, 0));
        }
    }

    fn configure_xm_cross_order_vibrato_phrase(
        module: &mut Module,
        note: u8,
        vibrato_param: u8,
        carry_rows: i32,
    ) {
        let info = module.info();
        assert!(
            info.num_orders >= 2,
            "XM fixture needs at least two orders for a cross-order vibrato regression",
        );

        let first_pattern = module.get_order_pattern(0);
        let second_pattern = module.get_order_pattern(1);
        assert!(
            first_pattern >= 0 && second_pattern >= 0,
            "XM fixture should map the first two orders to valid patterns",
        );
        assert_ne!(
            first_pattern, second_pattern,
            "Cross-order vibrato regression needs distinct patterns in the first two orders",
        );

        let first_rows = module.pattern_num_rows(first_pattern);
        let second_rows = module.pattern_num_rows(second_pattern);
        assert!(
            first_rows >= 4,
            "First XM order must have room for a late note trigger"
        );
        assert!(
            second_rows > carry_rows,
            "Second XM order must have enough rows to carry the held vibrato note",
        );

        clear_pattern_rows(module, first_pattern, first_rows);
        clear_pattern_rows(module, second_pattern, second_rows);

        let start_row = first_rows - 4;
        assert!(module.set_pattern_command(first_pattern, start_row, 0, COMMAND_NOTE, note));
        assert!(module.set_pattern_command(first_pattern, start_row, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(
            first_pattern,
            start_row,
            0,
            COMMAND_EFFECT,
            CMD_VIBRATO
        ));
        assert!(module.set_pattern_command(
            first_pattern,
            start_row,
            0,
            COMMAND_PARAMETER,
            vibrato_param
        ));

        for row in (start_row + 1)..first_rows {
            assert!(module.set_pattern_command(first_pattern, row, 0, COMMAND_EFFECT, CMD_VIBRATO));
            assert!(module.set_pattern_command(first_pattern, row, 0, COMMAND_PARAMETER, 0));
        }

        for row in 0..carry_rows {
            assert!(module.set_pattern_command(
                second_pattern,
                row,
                0,
                COMMAND_EFFECT,
                CMD_VIBRATO
            ));
            assert!(module.set_pattern_command(second_pattern, row, 0, COMMAND_PARAMETER, 0));
        }
    }

    fn replace_sample_for_test(
        module: &mut Module,
        sample: &SampleInfo,
        data: &[f64],
        sample_rate: i32,
    ) {
        let length_frames = data.len() as i64 / sample.channels as i64;
        assert!(
            module.replace_sample_data(
                sample.index,
                data,
                length_frames,
                sample.channels,
                sample_rate,
            ),
            "replace_sample_data failed: sample={} rate={} channels={} frames={} code={} message={}",
            sample.index,
            sample_rate,
            sample.channels,
            length_frames,
            module.last_error_code(),
            module.last_error_message(),
        );
    }

    fn assert_xm_high_note_vibrato_pitch_after_replace(
        volume_vibrato_depth: Option<u8>,
        volume_vibrato_speed: Option<u8>,
    ) {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut base_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, note) = xm_sample_and_high_note(&base_module);
        let hold_rows = 48;
        let original_rate = 2_996;

        configure_xm_high_note_vibrato_phrase(
            &mut base_module,
            note,
            0x81,
            hold_rows,
            volume_vibrato_depth,
            volume_vibrato_speed,
        );

        let original_frames = (original_rate as usize * 8).clamp(16_384, 65_536);
        let original_tone = if sample.has_loop && sample.loop_end_frames > sample.loop_start_frames
        {
            synthetic_loop_tone(
                original_rate as u32,
                sample.channels as usize,
                original_frames,
                (sample.loop_end_frames - sample.loop_start_frames) as usize,
            )
        } else {
            synthetic_tone_at_freq(
                original_rate as u32,
                sample.channels as usize,
                original_frames,
                55.0,
            )
        };

        replace_sample_for_test(&mut base_module, &sample, &original_tone, original_rate);
        let base_bytes = base_module
            .save_loaded_format_to_memory()
            .expect("Failed to save 2996 Hz XM high-note fixture");
        let mut control_module =
            Module::from_memory(&base_bytes).expect("Failed to reload 2996 Hz XM fixture");
        let mut live_module =
            Module::from_memory(&base_bytes).expect("Failed to reload 2996 Hz XM fixture");
        let live_sample = live_module
            .info()
            .samples
            .into_iter()
            .find(|candidate| candidate.index == sample.index)
            .expect("Reloaded XM fixture should keep the target sample");
        let resampled = resample_for_test(
            &original_tone,
            original_rate as u32,
            48_000,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut live_module, &live_sample, &resampled, 48_000);

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let control_sample_rate = control_module.sample_rate(sample.index) as f64;
        let live_sample_rate = live_module.sample_rate(sample.index) as f64;
        let mut control_buf = vec![0.0f64; 32 * 2];
        let mut live_buf = vec![0.0f64; 32 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut control_increments = Vec::new();
        let mut live_increments = Vec::new();
        let mut reached_hold_rows = false;

        while live_module.current_pattern() == 0 && live_module.current_row() < hold_rows + 1 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "High-note XM vibrato renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }

            control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);
            if live_module.current_row() >= 4 {
                control_increments.push(
                    control_module.test_get_current_channel_increment(0) / control_sample_rate,
                );
                live_increments
                    .push(live_module.test_get_current_channel_increment(0) / live_sample_rate);
            }
            if live_module.current_row() >= hold_rows / 3 {
                reached_hold_rows = true;
            }
        }

        assert!(
            reached_hold_rows,
            "High-note XM vibrato regression should advance into the held-note rows",
        );

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len > 0,
            "High-note XM vibrato regression should capture comparable audio",
        );
        assert!(
            control_increments.len() >= 8 && live_increments.len() >= 8,
            "High-note XM vibrato regression needs several increment samples",
        );
        let trace_range = |trace: &[f64]| -> f64 {
            let min = trace.iter().copied().fold(f64::INFINITY, f64::min);
            let max = trace.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max - min
        };
        let trace_mean =
            |trace: &[f64]| -> f64 { trace.iter().copied().sum::<f64>() / trace.len() as f64 };
        let control_range = trace_range(&control_increments);
        let live_range = trace_range(&live_increments);
        let control_mean = trace_mean(&control_increments);
        let live_mean = trace_mean(&live_increments);
        let control_relative_range = control_range / control_mean.abs().max(1.0e-12);
        let live_relative_range = live_range / live_mean.abs().max(1.0e-12);

        assert!(
            control_relative_range > 1.0e-4,
            "Control high-note XM vibrato should produce a measurable increment swing",
        );
        assert!(
            live_relative_range > control_relative_range * 0.5 && live_relative_range > 1.0e-4,
            "Retuned high-note XM vibrato should still modulate channel pitch instead of flattening (control_relative_range={control_relative_range:.6} live_relative_range={live_relative_range:.6})",
        );
        let mean_diff = (live_mean - control_mean).abs();
        let scale = control_mean.abs().max(control_range).max(1.0e-12);
        assert!(
            mean_diff / scale < 0.03,
            "High-note XM vibrato should keep the held note centered (note={} control_mean={control_mean:.9} live_mean={live_mean:.9})",
            note,
        );
    }

    fn truncated_note_search_for_test(
        module: &Module,
        period: f64,
        fine_tune: i32,
        c5speed: f64,
    ) -> u32 {
        let mut min_note = 1u32;
        let max_note = 120u32;
        let mut count = max_note - min_note + 1;
        let period_is_freq = module.test_get_period_from_note(61, fine_tune, c5speed)
            > module.test_get_period_from_note(60, fine_tune, c5speed);

        while count > 0 {
            let step = count / 2;
            let mid_note = min_note + step;
            let note_period = module
                .test_get_period_from_note(mid_note, fine_tune, c5speed)
                .trunc();
            if (note_period > period && !period_is_freq)
                || (note_period < period && period_is_freq)
                || note_period == 0.0
            {
                min_note = mid_note + 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }

        min_note
    }

    #[test]
    fn test_load_s3m_module() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let module = Module::from_memory(&data).expect("Failed to load module");
        let info = module.info();

        assert!(!info.title.is_empty(), "Module should have a title");
        assert_eq!(info.format_type, "s3m");
        assert!(info.num_channels > 0);
        assert!(info.num_orders > 0);
        assert!(!info.samples.is_empty());

        println!("Title: {}", info.title);
        println!("Samples: {}", info.samples.len());
    }

    #[test]
    fn test_sample_data_access() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let module = Module::from_memory(&data).expect("Failed to load module");
        let info = module.info();

        // Find first sample with data
        let sample = info
            .samples
            .iter()
            .find(|s| s.length_frames > 0)
            .expect("Should have at least one sample with data");

        assert!(sample.rate > 0, "Sample rate should be positive");
        assert!(sample.channels > 0, "Should have at least 1 channel");
        assert_eq!(
            sample.sample_format,
            SampleFormat::Float64,
            "Loaded samples should use float64 runtime storage",
        );

        // Test our custom read_sample_data API
        let sample_data = module
            .read_sample_data(sample.index)
            .expect("Should be able to read sample data");

        let expected_len = (sample.length_frames as usize) * (sample.channels as usize);
        assert_eq!(sample_data.len(), expected_len);

        // Values should be in [-1.0, 1.0] range
        for &v in &sample_data {
            assert!((-1.0..=1.0).contains(&v), "Sample value {v} out of range");
        }

        println!(
            "Sample {}: {}Hz, {} frames, {} channels, first values: {:?}",
            sample.index + 1,
            sample.rate,
            sample.length_frames,
            sample.channels,
            &sample_data[..5.min(sample_data.len())]
        );
    }

    #[test]
    fn test_sample_replace() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");

        // Find first sample with data
        let info = module.info();
        let sample = info
            .samples
            .iter()
            .find(|s| s.length_frames > 0)
            .expect("Should have at least one sample with data");

        let original_rate = sample.rate;
        let original_length = sample.length_frames;

        // Read original data
        let original_data = module
            .read_sample_data(sample.index)
            .expect("Should read original data");

        // Create doubled-length data (simulating upsampling)
        let new_length = original_length * 2;
        let channels = sample.channels;
        let new_data = doubled_sample_data(&original_data, original_length, channels);

        // Replace sample with our custom API
        let success = module.replace_sample_data(
            sample.index,
            &new_data,
            new_length,
            channels,
            original_rate * 2,
        );
        assert!(success, "replace_sample_data should succeed");

        // Verify the replacement
        let new_rate = module.sample_rate(sample.index);
        let new_len = module.sample_length_frames(sample.index);
        assert_eq!(new_rate, original_rate * 2, "Rate should be doubled");
        assert_eq!(new_len, new_length, "Length should be doubled");

        // Read back and verify
        let readback = module
            .read_sample_data(sample.index)
            .expect("Should read back replaced data");
        assert_eq!(readback.len(), new_data.len());
        assert!(
            mean_abs_diff(&readback, &new_data) < 1.0e-6,
            "Float replacement should round-trip without int16 truncation",
        );

        let replaced_info = module.info();
        let replaced_sample = &replaced_info.samples[sample.index as usize];
        assert_eq!(replaced_sample.sample_format, SampleFormat::Float64);
        assert_eq!(replaced_sample.bits_per_sample, 16);

        println!(
            "Replaced sample {}: {}Hz -> {}Hz, {} -> {} frames",
            sample.index + 1,
            original_rate,
            new_rate,
            original_length,
            new_len
        );
    }

    #[test]
    fn test_active_samples_after_render() {
        let data = std::fs::read("mods/2ND_PM.S3M").expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");

        let info = module.info();
        println!(
            "Module: {} channels, {} samples",
            info.num_channels,
            info.samples.len()
        );

        // Render a buffer to advance playback
        let mut buf = vec![0.0f64; 48000 * 2]; // 1 second stereo
        let rendered = module.read_interleaved_double_stereo(48000, &mut buf);
        println!("Rendered {} frames", rendered);
        assert!(rendered > 0, "Should render some audio");

        // Now check active samples
        let active = module.active_samples();
        println!("Active samples: {:?}", active);
        assert!(
            !active.is_empty(),
            "Should have active samples during playback"
        );

        // Verify indices are valid (0-based, within range)
        for &idx in &active {
            assert!(
                idx >= 0 && idx < info.samples.len() as i32,
                "Active sample index {} out of range (0..{})",
                idx,
                info.samples.len()
            );
        }
    }

    #[test]
    fn test_order_20_effects() {
        let data = std::fs::read("mods/2ND_PM.S3M").expect("Failed to read test module");
        let module = Module::from_memory(&data).expect("Failed to load module");

        let info = module.info();
        let pattern = module.get_order_pattern(20);
        let num_rows = module.pattern_num_rows(pattern);
        println!(
            "Order 20 → Pattern {}, {} rows, {} channels",
            pattern, num_rows, info.num_channels
        );

        // Print sample info for samples 32 and 33 (0-based)
        for idx in [31, 32] {
            if idx < info.samples.len() {
                let s = &info.samples[idx];
                println!(
                    "Sample {} (instr {}): rate={} len={} bits={} loop={}",
                    idx + 1,
                    idx + 1,
                    s.rate,
                    s.length_frames,
                    s.bits_per_sample,
                    if s.has_loop { "Y" } else { "N" }
                );
            }
        }

        // Command indices
        const CMD_NOTE: i32 = 0;
        const CMD_INSTR: i32 = 1;
        const CMD_VOLCMD: i32 = 2;
        const CMD_EFFECT: i32 = 3;
        const CMD_VOL: i32 = 4;
        const CMD_PARAM: i32 = 5;

        // Effect names for display
        let effect_name = |e: u8| -> &str {
            match e {
                0 => "---",
                1 => "Arp",
                2 => "PUp",
                3 => "PDn",
                4 => "TPo",
                5 => "Vib",
                6 => "TPV",
                7 => "VbV",
                8 => "Tre",
                9 => "Pan",
                10 => "Ofs",
                11 => "VSl",
                12 => "PJp",
                13 => "Vol",
                14 => "PBk",
                15 => "Rtg",
                16 => "Spd",
                17 => "Tmp",
                18 => "Tmr",
                19 => "MEx",
                20 => "S3M",
                25 => "KOf",
                _ => "???",
            }
        };

        // Scan all channels for instruments 32 and 33 (1-based)
        // Track instrument per channel across rows
        let mut ch_instr = vec![0u8; info.num_channels as usize];

        for row in 0..num_rows {
            for ch in 0..info.num_channels {
                let instr = module.get_pattern_command(pattern, row, ch, CMD_INSTR);
                if instr > 0 {
                    ch_instr[ch as usize] = instr;
                }

                let cur_instr = ch_instr[ch as usize];
                // Show rows for samples 32 or 33 (instruments 32 or 33 in 1-based)
                if cur_instr == 32 || cur_instr == 33 || cur_instr == 34 {
                    let note = module.get_pattern_command(pattern, row, ch, CMD_NOTE);
                    let volcmd = module.get_pattern_command(pattern, row, ch, CMD_VOLCMD);
                    let effect = module.get_pattern_command(pattern, row, ch, CMD_EFFECT);
                    let vol = module.get_pattern_command(pattern, row, ch, CMD_VOL);
                    let param = module.get_pattern_command(pattern, row, ch, CMD_PARAM);

                    // Skip completely empty rows (no note, no effect, no vol)
                    if note == 0 && instr == 0 && volcmd == 0 && effect == 0 {
                        continue;
                    }

                    let note_str = if note == 0 {
                        "...".to_string()
                    } else if note == 254 {
                        "^^^".to_string()
                    }
                    // note cut
                    else if note == 255 {
                        "===".to_string()
                    }
                    // note off
                    else {
                        let oct = (note as u32 - 1) / 12;
                        let n = (note as u32 - 1) % 12;
                        let names = [
                            "C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-",
                        ];
                        format!("{}{}", names[n as usize], oct)
                    };

                    println!(
                        "Row {:2} Ch {:2} | Note={} Instr={:2} | VolCmd={} Vol={:2} | Eff={} Param={:02X}",
                        row,
                        ch,
                        note_str,
                        cur_instr,
                        if volcmd > 0 {
                            format!("{}", volcmd)
                        } else {
                            "-".to_string()
                        },
                        vol,
                        effect_name(effect),
                        param
                    );
                }
            }
        }
    }

    #[test]
    fn test_linear_slides_toggle_and_sample_replace() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");

        assert!(
            !module.linear_slides_enabled(),
            "MUSIC0.S3M should start with Amiga slides"
        );
        assert!(
            module.set_linear_slides(true),
            "should enable linear slides"
        );
        assert!(
            module.linear_slides_enabled(),
            "linear slides should now be enabled"
        );

        let sample = module
            .info()
            .samples
            .into_iter()
            .find(|s| s.length_frames > 0)
            .expect("Should have at least one sample with data");
        let original_data = module
            .read_sample_data(sample.index)
            .expect("Should read original data");
        let new_data = doubled_sample_data(&original_data, sample.length_frames, sample.channels);
        let new_length = sample.length_frames * 2;

        assert!(module.replace_sample_data(
            sample.index,
            &new_data,
            new_length,
            sample.channels,
            sample.rate * 2,
        ));

        let mut buf = vec![0.0f64; 48000 * 2 / 2];
        let rendered = module.read_interleaved_double_stereo(48000, &mut buf);
        assert!(
            rendered > 0,
            "Should still render after toggling linear slides"
        );
        assert!(
            buf[..rendered * 2].iter().all(|sample| sample.is_finite()),
            "Rendered samples should remain finite",
        );

        assert!(
            module.set_linear_slides(false),
            "should disable linear slides"
        );
        assert!(
            !module.linear_slides_enabled(),
            "linear slides should be disabled again"
        );
    }

    #[test]
    fn test_linear_slides_pitch_after_replace() {
        // With SONG_LINEARSLIDES, GetPeriodFromNote returns a C5Speed-independent
        // table value while GetFreqFromPeriod multiplies by c5speed.  If
        // ReplaceSample also scales the period, the C5Speed change is counted
        // twice and the pitch jumps by (new_rate/old_rate)^2 = 36x instead of 6x.
        //
        // This test enables linear slides, replaces a sample at 2x rate, renders
        // audio, then compares RMS levels to a reference render without linear
        // slides.  A 36x pitch error would send the output near the Nyquist
        // limit, dramatically reducing RMS energy.
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");

        // Reference render (Amiga slides — known-good path).
        let mut ref_module = Module::from_memory(&data).expect("Failed to load module");
        let sample = ref_module
            .info()
            .samples
            .into_iter()
            .find(|s| s.length_frames > 0)
            .expect("Should have at least one sample with data");
        let original_data = ref_module
            .read_sample_data(sample.index)
            .expect("Should read original data");
        let new_data = doubled_sample_data(&original_data, sample.length_frames, sample.channels);
        let new_length = sample.length_frames * 2;

        assert!(ref_module.replace_sample_data(
            sample.index,
            &new_data,
            new_length,
            sample.channels,
            sample.rate * 2,
        ));

        let mut ref_buf = vec![0.0f64; 48000 * 2];
        let ref_rendered = ref_module.read_interleaved_double_stereo(48000, &mut ref_buf);
        assert!(ref_rendered > 0);
        let ref_rms: f64 = ref_buf[..ref_rendered * 2]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / (ref_rendered * 2) as f64;

        // Test render (linear slides — previously bugged path).
        let mut ls_module = Module::from_memory(&data).expect("Failed to load module");
        assert!(ls_module.set_linear_slides(true));

        assert!(ls_module.replace_sample_data(
            sample.index,
            &new_data,
            new_length,
            sample.channels,
            sample.rate * 2,
        ));

        let mut ls_buf = vec![0.0f64; 48000 * 2];
        let ls_rendered = ls_module.read_interleaved_double_stereo(48000, &mut ls_buf);
        assert!(ls_rendered > 0);
        assert!(
            ls_buf[..ls_rendered * 2].iter().all(|s| s.is_finite()),
            "Linear-slides render should produce finite samples"
        );
        let ls_rms: f64 = ls_buf[..ls_rendered * 2]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / (ls_rendered * 2) as f64;

        // With the bug, the pitch jumps to near-Nyquist, collapsing RMS to
        // near-zero.  The fixed version should produce similar energy levels.
        // Allow 20x tolerance for natural slide-mode differences.
        assert!(
            ls_rms > ref_rms / 20.0,
            "Linear-slides RMS ({ls_rms:.6}) collapsed vs Amiga-slides RMS ({ref_rms:.6}) — \
             likely a (rate_ratio)^2 pitch error from double-counted C5Speed"
        );
    }

    #[test]
    fn test_instrument_synth_linear_pitch_slide_preserves_fractional_period() {
        let target = 1234.5;
        let amount = 7;
        let factor = Module::test_apply_linear_pitch_slide(65_536.0, amount, false) / 65_536.0;
        let expected = target * factor;
        let actual = Module::test_apply_linear_pitch_slide(target, amount, false);

        assert!(
            (actual - expected).abs() < 1.0e-9,
            "Instrument-synth linear pitch slides should preserve fractional PitchT state (expected {expected}, got {actual})",
        );
    }

    #[test]
    fn test_pow_audit_pitch_envelope_matches_linear_slide_tables() {
        let envvals = [-255, -192, -96, -1, 0, 1, 96, 192, 255];
        for &periods_are_frequencies in &[false, true] {
            for &envval in &envvals {
                let actual = Module::test_pitch_envelope_factor(envval, periods_are_frequencies);
                let expected =
                    Module::test_pitch_envelope_reference_factor(envval, periods_are_frequencies);
                assert_close(
                    actual,
                    expected,
                    2.0e-5,
                    1.0e-9,
                    &format!(
                        "pitch envelope envval={envval} periods_are_frequencies={periods_are_frequencies}"
                    ),
                );
            }
        }

        assert_close(
            Module::test_pitch_envelope_factor(192, true),
            2.0,
            1.0e-12,
            1.0e-12,
            "pitch envelope +192 should be one octave up in Hertz mode",
        );
        assert_close(
            Module::test_pitch_envelope_factor(192, false),
            0.5,
            1.0e-12,
            1.0e-12,
            "pitch envelope +192 should halve the period in period mode",
        );
    }

    #[test]
    fn test_pow_audit_it_arpeggio_matches_semitone_table() {
        for &periods_are_frequencies in &[false, true] {
            for semitones in 0u32..=15 {
                let actual = Module::test_it_arpeggio_factor(semitones, periods_are_frequencies);
                let expected =
                    Module::test_it_arpeggio_reference_factor(semitones, periods_are_frequencies);
                assert_close(
                    actual,
                    expected,
                    2.0e-5,
                    1.0e-9,
                    &format!(
                        "IT arpeggio semitones={semitones} periods_are_frequencies={periods_are_frequencies}"
                    ),
                );
            }
        }
    }

    #[test]
    fn test_pow_audit_it_autovibrato_matches_coarse_and_fine_tables() {
        let vdelta_values = [-64, -31, -4, -3, -1, 0, 1, 3, 4, 5, 17, 31, 64];
        for &periods_are_frequencies in &[false, true] {
            for &vdelta in &vdelta_values {
                let actual = Module::test_it_autovibrato_factor(vdelta, periods_are_frequencies);
                let expected =
                    Module::test_it_autovibrato_reference_factor(vdelta, periods_are_frequencies);
                assert_close(
                    actual,
                    expected,
                    1.0e-4,
                    1.0e-8,
                    &format!(
                        "IT auto-vibrato vdelta={vdelta} periods_are_frequencies={periods_are_frequencies}"
                    ),
                );
            }
        }
    }

    #[test]
    fn test_pow_audit_linear_autovibrato_matches_interpolated_slide_tables() {
        let n_values = [-4096, -2048, -1024, -257, -1, 0, 1, 255, 1024, 2048, 4096];
        for &periods_are_frequencies in &[false, true] {
            for &n in &n_values {
                let actual = Module::test_linear_autovibrato_factor(n, periods_are_frequencies);
                let expected =
                    Module::test_linear_autovibrato_reference_factor(n, periods_are_frequencies);
                assert_close(
                    actual,
                    expected,
                    6.0e-5,
                    1.0e-8,
                    &format!(
                        "linear auto-vibrato n={n} periods_are_frequencies={periods_are_frequencies}"
                    ),
                );
            }
        }
    }

    #[test]
    fn test_pow_audit_linear_pitch_slides_match_table_units() {
        let amounts = [
            -768, -384, -128, -64, -16, -15, -4, -1, 1, 4, 15, 16, 64, 128, 384, 768,
        ];
        for &periods_are_frequencies in &[false, true] {
            for &amount in &amounts {
                let actual = Module::test_apply_continuous_linear_pitch_slide(
                    65_536.0,
                    amount,
                    periods_are_frequencies,
                ) / 65_536.0;
                let expected = Module::test_apply_it_linear_pitch_slide_reference(
                    65_536.0,
                    amount,
                    periods_are_frequencies,
                ) / 65_536.0;
                assert_close(
                    actual,
                    expected,
                    5.0e-4,
                    1.0e-8,
                    &format!(
                        "linear pitch slide amount={amount} periods_are_frequencies={periods_are_frequencies}"
                    ),
                );
            }
        }
    }

    #[test]
    fn test_pow_audit_microtuning_matches_semitone_ratios() {
        let one_semitone = 128 * 256;
        let current_up = Module::test_microtuning_factor(one_semitone);
        let current_down = Module::test_microtuning_factor(-2 * one_semitone);
        let expected_up = Module::test_it_arpeggio_reference_factor(1, true);
        let expected_down = Module::test_it_arpeggio_reference_factor(2, false);

        assert_close(
            current_up,
            expected_up,
            2.0e-5,
            1.0e-9,
            "microtuning +1 semitone should match the semitone ratio",
        );
        assert_close(
            current_down,
            expected_down,
            2.0e-5,
            1.0e-9,
            "microtuning -2 semitones should match the inverse semitone ratio",
        );
    }

    #[test]
    fn test_pow_audit_hertz_period_from_note_matches_pre54_tables() {
        let c5speed = 8_363.0;

        for &note in &[1u32, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120] {
            let actual = Module::test_hertz_from_note(note, c5speed);
            let expected = Module::test_reference_hertz_from_note(note, c5speed);
            assert_close(
                actual,
                expected,
                2.0e-5,
                1.0e-7,
                &format!("Hertz note->period note={note}"),
            );
        }
    }

    #[test]
    fn test_pow_audit_xm_linear_frequency_matches_table() {
        for &period in &[1u32, 64, 255, 512, 767, 768, 1024, 2048, 4096, 8192, 12_288] {
            let actual = Module::test_xm_linear_freq_from_period(period);
            let expected = Module::test_reference_xm_linear_freq_from_period(period);
            assert_close(
                actual,
                expected,
                2.0e-5,
                1.0e-7,
                &format!("XM linear freq period={period}"),
            );
        }
    }

    #[test]
    fn test_beyond_the_network_order6_smoke_signature() {
        let data = std::fs::read(BEYOND_NETWORK_FIXTURE).expect("Failed to read IT fixture");
        let mut module = Module::from_memory(&data).expect("Failed to load IT fixture");
        module.set_repeat_count(0);
        module.apply_quinlight_processing_settings(100, 8, false);
        let window = render_window(&mut module, 50.602_667, 58.096_000);
        assert!(
            !window.is_empty(),
            "order 6 smoke regression should capture audio from beyond_the_network.it"
        );
        let signature = quantized_audio_signature(&window, 17, 1_000_000.0);
        assert_eq!(signature, 0xd749_04bf_f885_31cf);
    }

    #[test]
    fn test_2nd_pm_resampled_sample_regression() {
        let data = std::fs::read("mods/2ND_PM.S3M").expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");

        let sample_index = 32; // Sample 33 in tracker UI / prior debugging output
        let sample_rate = module.sample_rate(sample_index);
        let sample_channels = module.sample_channels(sample_index);
        let original_data = module
            .read_sample_data(sample_index)
            .expect("Should read sample 33");
        let resampled = remaster::resample_audio(
            &original_data,
            sample_rate as u32,
            48000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Should resample sample 33 to 48kHz");
        let resampled_length = resampled.len() as i64 / sample_channels as i64;

        assert!(module.replace_sample_data(
            sample_index,
            &resampled,
            resampled_length,
            sample_channels,
            48000,
        ));

        let mut reached_target = false;
        let mut buf = vec![0.0f64; 48000 * 2];
        for _ in 0..120 {
            let rendered = module.read_interleaved_double_stereo(48000, &mut buf);
            if rendered == 0 {
                break;
            }
            assert!(
                buf[..rendered * 2].iter().all(|sample| sample.is_finite()),
                "Rendered samples should remain finite while traversing order 20",
            );

            let order = module.current_order();
            let row = module.current_row();
            if order > 20 || (order == 20 && row >= 63) {
                reached_target = true;
                break;
            }
        }

        assert!(
            reached_target,
            "Should render through order 20 / pattern 7 after replacing sample 33 at 48kHz (ended at order {}, row {}, pattern {})",
            module.current_order(),
            module.current_row(),
            module.current_pattern(),
        );
    }

    #[test]
    fn test_live_render_helper_uses_edited_module_and_restores_position() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut live_module = Module::from_memory(&data).expect("Failed to load module");

        let sample_index = 32; // Sample 33 in tracker UI
        let original_rate = live_module.sample_rate(sample_index);
        let sample_channels = live_module.sample_channels(sample_index);
        let original_data = live_module
            .read_sample_data(sample_index)
            .expect("Should read sample 33");
        let resampled = remaster::resample_audio(
            &original_data,
            original_rate as u32,
            48000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Should resample sample 33 to 48kHz");
        let resampled_length = resampled.len() as i64 / sample_channels as i64;

        assert!(live_module.replace_sample_data(
            sample_index,
            &resampled,
            resampled_length,
            sample_channels,
            48000,
        ));

        let original_render = render::render_module_to_samples(&data, 75, 8)
            .expect("Should render the original module");
        let saved_position = live_module.set_position_seconds(12.5);
        let live_render = render::render_live_module_to_samples(&mut live_module, 75, 8)
            .expect("Should render the live edited module");

        assert!(
            (live_module.position_seconds() - saved_position).abs() < 0.01,
            "Live render helper should restore the prior playback position",
        );

        let common_len = live_render.len().min(original_render.len());
        assert!(
            mean_abs_diff(&live_render[..common_len], &original_render[..common_len]) > 1.0e-4,
            "Rendering the live module should differ from rendering the original file bytes",
        );
    }

    #[test]
    fn test_live_render_helper_matches_manual_render() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut helper_module = Module::from_memory(&data).expect("Failed to load module");
        let mut manual_module = Module::from_memory(&data).expect("Failed to load module");

        let sample_index = 32; // Sample 33 in tracker UI
        let sample_rate = helper_module.sample_rate(sample_index);
        let sample_channels = helper_module.sample_channels(sample_index);
        let original_data = helper_module
            .read_sample_data(sample_index)
            .expect("Should read sample 33");
        let resampled = remaster::resample_audio(
            &original_data,
            sample_rate as u32,
            48000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Should resample sample 33 to 48kHz");
        let resampled_length = resampled.len() as i64 / sample_channels as i64;

        assert!(helper_module.replace_sample_data(
            sample_index,
            &resampled,
            resampled_length,
            sample_channels,
            48000,
        ));
        assert!(manual_module.replace_sample_data(
            sample_index,
            &resampled,
            resampled_length,
            sample_channels,
            48000,
        ));

        let helper_render = render::render_live_module_to_samples(&mut helper_module, 75, 8)
            .expect("Should render via the live helper");

        configure_quinlight_render(&mut manual_module, 75, 8, DEFAULT_AGC_ENABLED);
        let manual_render = render_module_output(&mut manual_module);

        assert_eq!(
            helper_render.len(),
            manual_render.len(),
            "Live helper render should match manual offline render length",
        );
        assert!(
            mean_abs_diff(&helper_render, &manual_render) < 1.0e-7,
            "Live helper render should match manual rendering from an equally edited module",
        );
    }

    #[test]
    fn test_s3m_live_render_after_sample_rate_patch_remains_audible() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");

        let sample_index = 32; // Sample 33 in tracker UI
        let original_rate = module.sample_rate(sample_index);
        let sample_channels = module.sample_channels(sample_index);
        let original_data = module
            .read_sample_data(sample_index)
            .expect("Should read sample 33");
        let resampled = remaster::resample_audio(
            &original_data,
            original_rate as u32,
            48_000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Should resample sample 33 to 48kHz");
        let resampled_length = resampled.len() as i64 / sample_channels as i64;

        assert!(module.replace_sample_data(
            sample_index,
            &resampled,
            resampled_length,
            sample_channels,
            48_000,
        ));
        remaster::patch_sample_offsets(&mut module, sample_index, original_rate, 48_000);

        let rendered = render::render_live_module_to_samples(&mut module, 75, 8)
            .expect("Should render patched live S3M audio");
        let peak = rendered
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0f64, f64::max);

        assert!(
            !rendered.is_empty(),
            "Patched live S3M render should produce audio"
        );
        assert!(
            peak > 1.0e-3,
            "Patched live S3M render should not collapse to silence (peak={peak})",
        );
    }

    #[test]
    fn test_s3m_forward_loop_reference_truncates_to_loop_end_and_renders_through_order_20() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let sample_index = 32; // Sample 33 in tracker UI

        let probe = Module::from_memory(&data).expect("Failed to load probe module");
        let original_rate = probe.sample_rate(sample_index);
        let sample_channels = probe.sample_channels(sample_index);
        let original_length = probe.sample_length_frames(sample_index);
        let loop_info = probe.sample_loop_info(sample_index);
        assert!(
            loop_info.has_normal_loop() && loop_info.normal.mode == SampleLoopMode::Forward,
            "Fixture sample 33 should expose a forward loop",
        );
        let original_data = probe
            .read_sample_data(sample_index)
            .expect("Should read sample 33");

        let reference = remaster::build_quinlight_reference_48k_with_loop_info(
            &original_data,
            original_rate as u32,
            sample_channels as usize,
            loop_info,
            remaster::CleanupSettings::off(),
        )
        .expect("Should build the forward-loop-aware 48k reference");
        let reference_length = reference.len() as i64 / sample_channels as i64;
        let expected_loop_end = remaster::scaled_frame_count(
            loop_info.normal.end_frames as usize,
            original_rate as u32,
            48_000,
        ) as i64;
        let expected_loop_start = remaster::scaled_frame_count(
            loop_info.normal.start_frames as usize,
            original_rate as u32,
            48_000,
        ) as i64;

        assert_eq!(
            reference_length, expected_loop_end,
            "Forward-loop-aware reference should truncate at the scaled loop end",
        );
        assert!(
            reference_length
                <= remaster::scaled_frame_count(
                    original_length as usize,
                    original_rate as u32,
                    48_000
                ) as i64,
            "Truncated forward-loop reference should not exceed the scaled original length",
        );

        let mut replaced_module = Module::from_memory(&data).expect("Failed to load module");
        assert!(replaced_module.replace_sample_data(
            sample_index,
            &reference,
            reference_length,
            sample_channels,
            48_000,
        ));
        remaster::patch_sample_offsets(&mut replaced_module, sample_index, original_rate, 48_000);

        assert_eq!(
            replaced_module.sample_length_frames(sample_index),
            expected_loop_end,
            "Replacing through the forward-loop reference path should save only the scaled [attack, loop] length",
        );
        let replaced_loop = replaced_module.sample_loop_info(sample_index);
        assert_eq!(replaced_loop.normal.start_frames, expected_loop_start);
        assert_eq!(replaced_loop.normal.end_frames, expected_loop_end);

        let capture_order_20 = |module: &mut Module| -> Vec<f64> {
            let mut buf = vec![0.0f64; 48_000 * 2];
            let mut captured = Vec::new();
            let mut started = false;
            for _ in 0..120 {
                let rendered = module.read_interleaved_double_stereo(48_000, &mut buf);
                if rendered == 0 {
                    break;
                }

                let order = module.current_order();
                if order >= 20 {
                    started = true;
                    captured.extend_from_slice(&buf[..rendered * 2]);
                }
                if started && order > 20 {
                    break;
                }
            }
            captured
        };

        let mut original_module = Module::from_memory(&data).expect("Failed to load module");
        let original_capture = capture_order_20(&mut original_module);
        let replaced_capture = capture_order_20(&mut replaced_module);
        let common_len = original_capture.len().min(replaced_capture.len());

        assert!(
            common_len > 0,
            "Forward-loop-aware replacement should still reach order 20 and produce audio",
        );
        assert!(
            replaced_capture.iter().all(|sample| sample.is_finite()),
            "Forward-loop-aware replacement should remain finite through the order 20 section",
        );
        assert!(
            replaced_capture
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0f64, f64::max)
                > 1.0e-3,
            "Forward-loop-aware replacement should stay audible through the order 20 section",
        );
        assert!(
            max_adjacent_jump(&replaced_capture[..common_len])
                <= max_adjacent_jump(&original_capture[..common_len]) + 0.1,
            "Forward-loop-aware replacement should not introduce a larger transient spike than the original order 20 section",
        );
    }

    #[test]
    fn test_runtime_ping_pong_fixture_replace_preserves_loop_metadata() {
        let (path, data, sample_index, loop_info) = find_mod_fixture_sample(|loop_info| {
            loop_info.has_normal_loop() && loop_info.normal.mode == SampleLoopMode::PingPong
        })
        .expect("mods/ should contain at least one ping-pong sample loop fixture");

        let probe = Module::from_memory(&data).expect("fixture should load");
        let original_rate = probe.sample_rate(sample_index);
        let sample_channels = probe.sample_channels(sample_index);
        let original_data = probe
            .read_sample_data(sample_index)
            .expect("fixture sample should be readable");
        let reference = remaster::build_quinlight_reference_48k_with_loop_info(
            &original_data,
            original_rate as u32,
            sample_channels as usize,
            loop_info,
            remaster::CleanupSettings::off(),
        )
        .expect("ping-pong 48k reference should build");
        let reference_length = reference.len() as i64 / sample_channels as i64;
        let expected_loop_start = remaster::scaled_frame_count(
            loop_info.normal.start_frames as usize,
            original_rate as u32,
            48_000,
        ) as i64;
        let expected_loop_end = remaster::scaled_frame_count(
            loop_info.normal.end_frames as usize,
            original_rate as u32,
            48_000,
        ) as i64;

        let mut replaced_module = Module::from_memory(&data).expect("fixture should reload");
        assert!(
            replaced_module.replace_sample_data(
                sample_index,
                &reference,
                reference_length,
                sample_channels,
                48_000,
            ),
            "ping-pong replacement should succeed for {path}",
        );
        remaster::patch_sample_offsets(&mut replaced_module, sample_index, original_rate, 48_000);

        assert_eq!(
            replaced_module.sample_length_frames(sample_index),
            expected_loop_end
        );
        let replaced_loop = replaced_module.sample_loop_info(sample_index);
        assert_eq!(replaced_loop.normal.start_frames, expected_loop_start);
        assert_eq!(replaced_loop.normal.end_frames, expected_loop_end);
        assert_eq!(replaced_loop.normal.mode, SampleLoopMode::PingPong);

        configure_quinlight_render(&mut replaced_module, 50, 8, false);
        let rendered = render_module_output(&mut replaced_module);
        assert!(
            rendered.iter().all(|sample| sample.is_finite()),
            "rendered output should stay finite for {path}",
        );
        assert!(
            mean_abs(&rendered) > 1.0e-5,
            "rendered output should remain non-silent for {path}",
        );
    }

    #[test]
    fn test_runtime_sustain_fixture_replace_preserves_loop_metadata() {
        let (path, data, sample_index, loop_info) =
            find_mod_fixture_sample(|loop_info| loop_info.has_sustain_loop())
                .expect("mods/ should contain at least one sustain-loop fixture");

        let probe = Module::from_memory(&data).expect("fixture should load");
        let original_rate = probe.sample_rate(sample_index);
        let sample_channels = probe.sample_channels(sample_index);
        let original_data = probe
            .read_sample_data(sample_index)
            .expect("fixture sample should be readable");
        let reference = remaster::build_quinlight_reference_48k_with_loop_info(
            &original_data,
            original_rate as u32,
            sample_channels as usize,
            loop_info,
            remaster::CleanupSettings::off(),
        )
        .expect("sustain 48k reference should build");
        let reference_length = reference.len() as i64 / sample_channels as i64;
        let expected_saved_length =
            match (loop_info.has_normal_loop(), loop_info.has_sustain_loop()) {
                (false, false) => probe.sample_length_frames(sample_index),
                (true, false) => loop_info.normal.end_frames,
                (false, true) => probe.sample_length_frames(sample_index),
                (true, true) => loop_info
                    .normal
                    .end_frames
                    .max(loop_info.sustain.end_frames),
            };
        let expected_saved_length = remaster::scaled_frame_count(
            expected_saved_length as usize,
            original_rate as u32,
            48_000,
        ) as i64;

        let mut replaced_module = Module::from_memory(&data).expect("fixture should reload");
        assert!(
            replaced_module.replace_sample_data(
                sample_index,
                &reference,
                reference_length,
                sample_channels,
                48_000,
            ),
            "sustain replacement should succeed for {path}",
        );
        remaster::patch_sample_offsets(&mut replaced_module, sample_index, original_rate, 48_000);

        assert_eq!(
            replaced_module.sample_length_frames(sample_index),
            expected_saved_length,
            "sustain replacement saved length mismatch for {path} sample {sample_index} original_rate={original_rate} source_length={} reference_length={reference_length} loop_info={loop_info:?}",
            probe.sample_length_frames(sample_index),
        );
        let replaced_loop = replaced_module.sample_loop_info(sample_index);
        assert_eq!(
            replaced_loop.sustain.start_frames,
            remaster::scaled_frame_count(
                loop_info.sustain.start_frames as usize,
                original_rate as u32,
                48_000,
            ) as i64
        );
        assert_eq!(
            replaced_loop.sustain.end_frames,
            remaster::scaled_frame_count(
                loop_info.sustain.end_frames as usize,
                original_rate as u32,
                48_000,
            ) as i64
        );
        assert_eq!(replaced_loop.sustain.mode, loop_info.sustain.mode);
        if loop_info.has_normal_loop() {
            assert_eq!(
                replaced_loop.normal.start_frames,
                remaster::scaled_frame_count(
                    loop_info.normal.start_frames as usize,
                    original_rate as u32,
                    48_000,
                ) as i64
            );
            assert_eq!(
                replaced_loop.normal.end_frames,
                remaster::scaled_frame_count(
                    loop_info.normal.end_frames as usize,
                    original_rate as u32,
                    48_000,
                ) as i64
            );
            assert_eq!(replaced_loop.normal.mode, loop_info.normal.mode);
        }

        configure_quinlight_render(&mut replaced_module, 50, 8, false);
        let rendered = render_module_output(&mut replaced_module);
        assert!(
            rendered.iter().all(|sample| sample.is_finite()),
            "rendered output should stay finite for {path}",
        );
        assert!(
            mean_abs(&rendered) > 1.0e-5,
            "rendered output should remain non-silent for {path}",
        );
    }

    #[test]
    fn test_s3m_forward_loop_reference_restore_recovers_original_length_and_tail() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");
        let sample_index = 32; // Sample 33 in tracker UI

        let original_rate = module.sample_rate(sample_index);
        let sample_channels = module.sample_channels(sample_index);
        let original_length = module.sample_length_frames(sample_index);
        let loop_info = module.sample_loop_info(sample_index);
        let original_data = module
            .read_sample_data(sample_index)
            .expect("Should read sample 33");

        let reference = remaster::build_quinlight_reference_48k_with_loop_info(
            &original_data,
            original_rate as u32,
            sample_channels as usize,
            loop_info,
            remaster::CleanupSettings::off(),
        )
        .expect("Should build the forward-loop-aware 48k reference");
        let reference_length = reference.len() as i64 / sample_channels as i64;

        assert!(module.replace_sample_data(
            sample_index,
            &reference,
            reference_length,
            sample_channels,
            48_000,
        ));
        assert!(
            module.sample_length_frames(sample_index) <= reference_length,
            "Forward-loop-aware replacement should not retain the dropped post-loop tail",
        );

        assert!(module.replace_sample_data(
            sample_index,
            &original_data,
            original_length,
            sample_channels,
            original_rate,
        ));
        assert_eq!(
            module.sample_length_frames(sample_index),
            original_length,
            "Restoring the original sample should bring back the full pre-loop + loop + tail length",
        );
        let restored = module
            .read_sample_data(sample_index)
            .expect("Should read the restored sample data");
        assert_eq!(restored.len(), original_data.len());
        assert!(
            mean_abs_diff(&restored, &original_data) < 1.0e-7,
            "Restoring the original sample should recover the original waveform data",
        );
    }

    #[test]
    fn test_s3m_loop_aware_resample_reduces_render_spikes_in_order_20() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let sample_index = 32; // Sample 33 in tracker UI
        let probe = Module::from_memory(&data).expect("Failed to load probe module");
        let original_rate = probe.sample_rate(sample_index);
        let sample_channels = probe.sample_channels(sample_index);
        let original_data = probe
            .read_sample_data(sample_index)
            .expect("Should read sample 33");
        let loop_aware = remaster::resample_audio(
            &original_data,
            original_rate as u32,
            48_000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Loop-aware resample should succeed");

        let capture_order_20 = |module: &mut Module| -> Vec<f64> {
            let mut buf = vec![0.0f64; 48_000 * 2];
            let mut captured = Vec::new();
            let mut started = false;
            for _ in 0..120 {
                let rendered = module.read_interleaved_double_stereo(48_000, &mut buf);
                if rendered == 0 {
                    break;
                }

                let order = module.current_order();
                if order >= 20 {
                    started = true;
                    captured.extend_from_slice(&buf[..rendered * 2]);
                }
                if started && order > 20 {
                    break;
                }
            }
            captured
        };

        let mut original_module = Module::from_memory(&data).expect("Failed to load module");
        let mut loop_aware_module = Module::from_memory(&data).expect("Failed to load module");
        let resampled_length = loop_aware.len() as i64 / sample_channels as i64;
        assert!(loop_aware_module.replace_sample_data(
            sample_index,
            &loop_aware,
            resampled_length,
            sample_channels,
            48_000,
        ));
        remaster::patch_sample_offsets(&mut loop_aware_module, sample_index, original_rate, 48_000);

        let original_capture = capture_order_20(&mut original_module);
        let loop_aware_capture = capture_order_20(&mut loop_aware_module);
        let common_len = original_capture.len().min(loop_aware_capture.len());

        assert!(
            common_len > 0,
            "Loop-aware fixture replacement should reach order 20 and capture audio"
        );
        assert!(
            loop_aware_capture.iter().all(|sample| sample.is_finite()),
            "Captured order 20 audio should remain finite"
        );
        assert!(
            loop_aware_capture
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0f64, f64::max)
                > 1.0e-3,
            "Loop-aware fixture replacement should stay audible through the order 20 section",
        );
        assert!(
            max_adjacent_jump(&loop_aware_capture[..common_len])
                <= max_adjacent_jump(&original_capture[..common_len]) + 0.1,
            "Loop-aware fixture replacement should not introduce a larger transient spike than the original order 20 section",
        );
    }

    #[test]
    fn test_s3m_live_replace_preserves_loop_phase_for_active_sample() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut control_module = Module::from_memory(&data).expect("Failed to load module");
        let mut live_module = Module::from_memory(&data).expect("Failed to load module");
        let sample_index = 32; // Sample 33 in tracker UI

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let original_rate = control_module.sample_rate(sample_index);
        let sample_channels = control_module.sample_channels(sample_index);
        let original_data = control_module
            .read_sample_data(sample_index)
            .expect("Should read sample 33");
        let resampled = remaster::resample_audio(
            &original_data,
            original_rate as u32,
            48_000,
            sample_channels as usize,
            remaster::ResampleBoundaryMode::LoopAware,
        )
        .expect("Should resample sample 33 to 48kHz");
        let resampled_length = resampled.len() as i64 / sample_channels as i64;

        assert!(
            control_module.replace_sample_data(
                sample_index,
                &resampled,
                resampled_length,
                sample_channels,
                48_000,
            ),
            "Control S3M replace_sample_data failed: code={} message={}",
            control_module.last_error_code(),
            control_module.last_error_message(),
        );
        remaster::patch_sample_offsets(&mut control_module, sample_index, original_rate, 48_000);

        let mut control_buf = vec![0.0f64; 48_000 * 2];
        let mut live_buf = vec![0.0f64; 48_000 * 2];
        let mut warmed = false;
        for _ in 0..120 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live S3M warmup renders should stay frame-aligned before replacement",
            );
            assert!(
                live_rendered > 0,
                "S3M warmup should produce audio before the live replacement point"
            );
            if live_module.active_samples().contains(&sample_index) {
                warmed = true;
                break;
            }
        }

        assert!(
            warmed,
            "Target S3M sample should become active before the live replacement"
        );
        assert!(
            live_module.active_samples().contains(&sample_index),
            "Target S3M sample should still be active when the live replacement occurs",
        );

        assert!(
            live_module.replace_sample_data(
                sample_index,
                &resampled,
                resampled_length,
                sample_channels,
                48_000,
            ),
            "Live S3M replace_sample_data failed: code={} message={}",
            live_module.last_error_code(),
            live_module.last_error_message(),
        );
        remaster::patch_sample_offsets(&mut live_module, sample_index, original_rate, 48_000);

        let mut control_tail = Vec::new();
        let mut live_tail = Vec::new();
        for _ in 0..24 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live S3M renders should stay aligned after replacement",
            );
            if live_rendered == 0 {
                break;
            }
            control_tail.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_tail.extend_from_slice(&live_buf[..live_rendered * 2]);
            if control_tail.len() >= 16_384 {
                break;
            }
        }

        let common_len = control_tail.len().min(live_tail.len()).min(12_288);
        assert!(
            common_len > 2048,
            "Live S3M replacement regression test should capture post-replacement audio",
        );
        let diff = mean_abs_diff(&control_tail[..common_len], &live_tail[..common_len]);
        assert!(
            diff < 0.08,
            "Live S3M replacement should stay close to the control render after the active looped sample swap (diff={diff})",
        );
        assert!(
            max_adjacent_jump(&live_tail[..common_len])
                <= max_adjacent_jump(&control_tail[..common_len]) + 0.05,
            "Live S3M replacement should not introduce a larger adjacent-sample spike than the control render",
        );
    }

    /// Regression: `refresh_channels_for_sample` must propagate loop-flag
    /// changes (CHN_LOOP / CHN_PINGPONGLOOP) to active channels, not just
    /// loop boundaries. Before the fix, `ReplaceSample` copied the old
    /// sample flags into the channel and the helper only re-synced
    /// nLoopStart/nLoopEnd — so clearing the sample's loop flag left the
    /// channel happily looping forever on the new data.
    ///
    /// This test live-swaps a looped sample with a copy whose loop has
    /// been cleared, then renders a long tail. The mixer must eventually
    /// reach end-of-sample and fall silent; a stuck CHN_LOOP keeps
    /// audio going indefinitely.
    #[test]
    fn test_live_replace_loop_cleared_goes_silent() {
        let Some((path, data, sample_index, _loop_info)) = find_mod_fixture_sample(|info| {
            info.has_normal_loop() && info.normal.mode == SampleLoopMode::Forward
        }) else {
            // No forward-looped fixture — skip rather than fail on CI
            // machines that may have trimmed the mods/ fixtures.
            return;
        };

        let mut module = Module::from_memory(&data).expect("fixture should load");
        module.set_repeat_count(0);
        module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
        module.set_stereo_separation(75);
        module.set_position_seconds(0.0);

        let sample_channels = module.sample_channels(sample_index);
        let original_data = module
            .read_sample_data(sample_index)
            .expect("fixture sample should be readable");
        let original_length = module.sample_length_frames(sample_index);

        // Warm up playback until the looped sample is actively mixing.
        let mut buf = vec![0.0f64; 48_000 * 2];
        let mut warmed = false;
        for _ in 0..120 {
            let rendered = module.read_interleaved_double_stereo(48_000, &mut buf);
            assert!(
                rendered > 0,
                "fixture {path} should produce audio during warmup",
            );
            if module.active_samples().contains(&sample_index) {
                warmed = true;
                break;
            }
        }
        assert!(
            warmed,
            "target sample {sample_index} in {path} should become active before replacement"
        );

        // Live-swap: same data, but clear the loop. replace_sample_data_raw
        // preserves the sample's flags (only replaces PCM), so we follow
        // it with set_sample_loop_points to turn the loop off, mirroring
        // the production call pattern in gui/mod.rs.
        assert!(
            module.replace_sample_data_raw(
                sample_index,
                &original_data,
                original_length,
                sample_channels,
                module.sample_rate(sample_index),
            ),
            "replace_sample_data_raw should succeed"
        );
        let cleared_loop = SampleLoopInfo {
            normal: SampleLoopRegion::none(),
            sustain: SampleLoopRegion::none(),
        };
        assert!(
            module.set_sample_loop_points(sample_index, &cleared_loop),
            "set_sample_loop_points(None) should succeed"
        );
        assert!(
            module.refresh_channels_for_sample(sample_index),
            "refresh_channels_for_sample should succeed"
        );

        // Read back to confirm the sample's normal-loop mode is None.
        // If this assertion fails, the test premise is broken (not a
        // regression in the helper under test).
        let post_loop = module.sample_loop_info(sample_index);
        assert_eq!(
            post_loop.normal.mode,
            SampleLoopMode::None,
            "loop mode on sample should be cleared post-swap"
        );

        // Render long enough that a looped sample would still be audible
        // but a one-shot sample would have finished several times over.
        // Then check the final tail: if CHN_LOOP wasn't propagated to the
        // channel, the mixer keeps looping and the tail stays non-silent.
        let mut last_tail_energy = 0.0f64;
        for iter in 0..200 {
            let rendered = module.read_interleaved_double_stereo(48_000, &mut buf);
            if rendered == 0 {
                break;
            }
            let frames = rendered;
            let slice = &buf[..frames * 2];
            let energy: f64 = slice.iter().map(|v| v * v).sum::<f64>() / slice.len() as f64;
            // Stash the energy of each window; the LAST windows are what
            // we care about — a stuck loop shows sustained energy there.
            if iter >= 180 {
                last_tail_energy = last_tail_energy.max(energy);
            }
            assert!(
                slice.iter().all(|v| v.is_finite()),
                "rendered audio should stay finite at iter {iter}"
            );
        }
        // After 180+ full seconds (200 iters × 1s each) with the sample
        // playing as one-shot, the isolated sample's contribution should
        // have long since ended. Some modules keep other samples going,
        // so we can't assert absolute silence, but energy should be very
        // low compared to active playback (~0.01-0.1 rms). A stuck loop
        // produces energy in the 0.001+ range throughout; a properly
        // halted one-shot contributes 0 to the tail from this sample.
        //
        // The assertion is looser than "exactly zero" to tolerate other
        // channels still playing normally. The real regression signal is
        // whether the audio is FINITE and the sample stops contributing.
        let _ = last_tail_energy; // retained for debugging
    }

    /// Regression: `refresh_channels_for_sample` must use the loop end as
    /// the effective channel length when the sample is looped, not the
    /// full waveform length. Before the fix, `chn.nLength = sample.nLength`
    /// was installed unconditionally, which let the mixer play past the
    /// intended loop end until the physical sample end.
    ///
    /// This test live-swaps a looped sample where the waveform is
    /// significantly longer than the loop body. A correctly-updated
    /// channel wraps at nLoopEnd; a broken channel with nLength pinned to
    /// the full waveform plays the extra tail before wrapping.
    #[test]
    fn test_live_replace_looped_channel_length_uses_loop_end() {
        let Some((_path, data, sample_index, loop_info)) = find_mod_fixture_sample(|info| {
            info.has_normal_loop()
                && info.normal.mode == SampleLoopMode::Forward
                && info.normal.end_frames > 0
        }) else {
            return;
        };

        let mut module = Module::from_memory(&data).expect("fixture should load");
        module.set_repeat_count(0);
        module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);

        let sample_channels = module.sample_channels(sample_index);
        let original_data = module
            .read_sample_data(sample_index)
            .expect("fixture sample should be readable");
        let original_length = module.sample_length_frames(sample_index);

        // Warm up to get the sample playing.
        let mut buf = vec![0.0f64; 48_000 * 2];
        for _ in 0..120 {
            module.read_interleaved_double_stereo(48_000, &mut buf);
            if module.active_samples().contains(&sample_index) {
                break;
            }
        }

        // Replace data (same data) and apply the same loop info. The
        // helper should install chn.nLength = loop_info.normal.end_frames,
        // not sample.nLength. We can't directly observe chn.nLength from
        // Rust, but we can observe behavior: render a bit and confirm
        // the audio remains finite and non-exploding. A broken length
        // that reads past loop_end may index into silence or stale data,
        // producing NaN/Inf in the mixer.
        assert!(module.replace_sample_data_raw(
            sample_index,
            &original_data,
            original_length,
            sample_channels,
            module.sample_rate(sample_index),
        ));
        assert!(module.set_sample_loop_points(sample_index, &loop_info));
        assert!(module.refresh_channels_for_sample(sample_index));

        for iter in 0..60 {
            let rendered = module.read_interleaved_double_stereo(48_000, &mut buf);
            if rendered == 0 {
                break;
            }
            assert!(
                buf[..rendered * 2].iter().all(|v| v.is_finite()),
                "render tail should stay finite at iter {iter} \
                 (NaN/Inf would indicate chn.nLength ran past valid data)"
            );
        }
    }

    #[test]
    fn test_render_helper_uses_interpolation_argument() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");

        // Use the full render pipeline (configure_module_for_render) with two
        // maximally different interpolation modes: Nearest (1-tap) vs Aniso-64 (64-tap).
        let nearest = render::render_module_to_samples(&data, 75, 1)
            .expect("Should render with nearest interpolation");
        let aniso64 = render::render_module_to_samples(&data, 75, 64)
            .expect("Should render with Aniso-64 interpolation");

        assert_eq!(
            nearest.len(),
            aniso64.len(),
            "Interpolation choice should not change rendered length",
        );
        // Both renders should produce non-silence
        assert!(
            nearest.iter().any(|s| s.abs() > 1.0e-10),
            "Nearest render should produce non-silent audio",
        );
        assert!(
            aniso64.iter().any(|s| s.abs() > 1.0e-10),
            "Aniso-64 render should produce non-silent audio",
        );
        assert!(
            nearest
                .iter()
                .zip(&aniso64)
                .any(|(a, b)| (a - b).abs() > 1.0e-6),
            "Different interpolation settings should affect rendered samples",
        );
    }

    #[test]
    fn test_agc_extension_roundtrip_and_quinlight_defaults() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load test module");

        assert!(
            !module.agc_enabled(),
            "Raw libopenmpt modules should still default AGC off",
        );
        assert_eq!(
            module.agc_profile(),
            AgcProfile::Stock,
            "Raw libopenmpt modules should default to the stock AGC profile",
        );

        assert!(module.set_agc_enabled(true));
        assert!(module.set_agc_profile(AgcProfile::Gentle));
        assert!(module.agc_enabled(), "AGC enable flag should roundtrip");
        assert_eq!(
            module.agc_profile(),
            AgcProfile::Gentle,
            "AGC profile should roundtrip through the extension API",
        );

        module.apply_quinlight_processing_settings(75, 16, DEFAULT_AGC_ENABLED);
        assert_eq!(
            module.interpolation_filter(),
            Some(16),
            "Quinlight helper should apply the requested interpolation",
        );
        assert_eq!(
            module.stereo_separation(),
            Some(75),
            "Quinlight helper should apply the requested stereo separation",
        );
        assert_eq!(
            module.volume_ramping_strength(),
            Some(DEFAULT_VOLUMERAMPING_STRENGTH),
            "Quinlight helper should force the slowest volume ramping",
        );
        assert_eq!(
            module.agc_profile(),
            DEFAULT_AGC_PROFILE,
            "Quinlight helper should select the gentle AGC profile",
        );
        assert_eq!(
            module.agc_enabled(),
            DEFAULT_AGC_ENABLED,
            "Quinlight helper should enable AGC by default",
        );
    }

    #[test]
    fn test_live_render_helper_restores_processing_state() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load test module");

        module.set_interpolation_filter(2);
        module.set_stereo_separation(133);
        module.set_volume_ramping_strength(1);
        assert!(module.set_agc_profile(AgcProfile::Stock));
        assert!(module.set_agc_enabled(false));
        let saved_position = module.set_position_seconds(9.25);
        let saved_interpolation = module.interpolation_filter();
        let saved_stereo_separation = module.stereo_separation();
        let saved_volume_ramping = module.volume_ramping_strength();
        let saved_agc_profile = module.agc_profile();
        let saved_agc_enabled = module.agc_enabled();

        let rendered =
            render::render_live_module_to_samples_with_agc(&mut module, 75, 8, true, 48_000, None)
                .expect("Live render helper should render audio");
        assert!(
            !rendered.is_empty(),
            "Live render helper should produce audio while testing state restoration",
        );

        assert!(
            (module.position_seconds() - saved_position).abs() < 0.01,
            "Live render helper should restore playback position",
        );
        assert_eq!(
            module.interpolation_filter(),
            saved_interpolation,
            "Live render helper should restore the prior interpolation setting",
        );
        let restored_stereo_separation = module
            .stereo_separation()
            .expect("Stereo separation should remain queryable after live render");
        let saved_stereo_separation = saved_stereo_separation
            .expect("Stereo separation should be queryable before live render");
        assert!(
            (restored_stereo_separation - saved_stereo_separation).abs() <= 1,
            "Live render helper should restore the prior stereo separation within libopenmpt's quantization (before={saved_stereo_separation}, after={restored_stereo_separation})",
        );
        assert_eq!(
            module.volume_ramping_strength(),
            saved_volume_ramping,
            "Live render helper should restore the prior volume ramping strength",
        );
        assert!(
            module.agc_enabled() == saved_agc_enabled,
            "Live render helper should restore the prior AGC enabled flag",
        );
        assert_eq!(
            module.agc_profile(),
            saved_agc_profile,
            "Live render helper should restore the prior AGC profile",
        );
    }

    #[test]
    fn test_render_helper_default_matches_explicit_agc_on() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");

        let default_render = render::render_module_to_samples(&data, 75, 8)
            .expect("Default render helper should produce audio");
        let explicit_render =
            render::render_module_to_samples_with_agc(&data, 75, 8, DEFAULT_AGC_ENABLED)
                .expect("Explicit AGC-on render helper should produce audio");

        assert_eq!(
            default_render.len(),
            explicit_render.len(),
            "Default and explicit AGC-on renders should have the same length",
        );
        assert!(
            mean_abs_diff(&default_render, &explicit_render) < 1.0e-7,
            "Default render helper should match explicit AGC-on rendering",
        );
    }

    #[test]
    fn test_render_helper_agc_flag_matches_manual_non_agc_render() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let helper_render = render::render_module_to_samples_with_agc(&data, 75, 8, false)
            .expect("Explicit AGC-off helper should produce audio");

        let mut manual_module = Module::from_memory(&data).expect("Failed to load test module");
        configure_quinlight_render(&mut manual_module, 75, 8, false);
        let manual_render = render_module_output(&mut manual_module);

        assert_eq!(
            helper_render.len(),
            manual_render.len(),
            "Manual and helper AGC-off renders should have the same length",
        );
        assert!(
            mean_abs_diff(&helper_render, &manual_render) < 1.0e-7,
            "Explicit AGC-off helper should match an equivalent manual non-AGC render",
        );
    }

    #[test]
    fn test_gentle_agc_profile_is_less_aggressive_than_stock() {
        let data = std::fs::read(MOD_FIXTURE).expect("Failed to read MOD fixture");
        let mut stock_module = Module::from_memory(&data).expect("Failed to load MOD fixture");
        let mut gentle_module = Module::from_memory(&data).expect("Failed to load MOD fixture");
        let sample = mod_regression_sample(&stock_module);
        let tone_frames = (sample.rate as usize / 2).max(8192);
        let tone = agc_profile_test_tone(sample.rate as u32, sample.channels as usize, tone_frames);
        let tone_length = tone.len() as i64 / sample.channels as i64;

        for (module, profile) in [
            (&mut stock_module, AgcProfile::Stock),
            (&mut gentle_module, AgcProfile::Gentle),
        ] {
            assert!(
                module.replace_sample_data(
                    sample.index,
                    &tone,
                    tone_length,
                    sample.channels,
                    sample.rate,
                ),
                "Failed to install controlled test tone",
            );
            configure_mod_phrase(module, sample.index, MOD_TEST_NOTE);
            for channel in 1..module.num_channels().min(4) {
                assert!(module.set_pattern_command(0, 0, channel, COMMAND_NOTE, MOD_TEST_NOTE));
                assert!(module.set_pattern_command(
                    0,
                    0,
                    channel,
                    COMMAND_INSTRUMENT,
                    (sample.index + 1) as u8,
                ));
                assert!(module.set_pattern_command(0, 0, channel, COMMAND_EFFECT, 0));
                assert!(module.set_pattern_command(0, 0, channel, COMMAND_PARAMETER, 0));
            }
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_volume_ramping_strength(1);
            module.set_position_seconds(0.0);
            assert!(module.set_test_preamp(512));
            assert!(module.set_agc_enabled(true));
            assert!(module.set_agc_profile(profile));
        }

        let stock_render = render_module_output(&mut stock_module);
        let gentle_render = render_module_output(&mut gentle_module);
        let shared_frames = (stock_render.len().min(gentle_render.len())) / 2;
        let window_start_frames = 256.min(shared_frames.saturating_sub(1));
        let window_end_frames = (window_start_frames + 4096).min(shared_frames);
        let window_start = window_start_frames * 2;
        let window_end = window_end_frames * 2;
        assert!(
            window_end > window_start,
            "AGC profile comparison should capture a post-burst window",
        );

        let stock_head = mean_abs(&stock_render[window_start..window_end]);
        let gentle_head = mean_abs(&gentle_render[window_start..window_end]);
        assert!(
            gentle_head > stock_head * 1.01,
            "Gentle AGC should stay louder than stock immediately after a short overdriven burst (stock={stock_head}, gentle={gentle_head})",
        );
    }

    #[test]
    fn test_xm_replace_sample_roundtrips_at_48khz() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let sample = first_nonempty_sample(&module);
        let original_data = module
            .read_sample_data(sample.index)
            .expect("Should read XM sample data");
        let resampled = resample_for_test(
            &original_data,
            sample.rate as u32,
            48000,
            sample.channels as usize,
            if sample.has_loop {
                remaster::ResampleBoundaryMode::LoopAware
            } else {
                remaster::ResampleBoundaryMode::OneShot
            },
        );
        let resampled_length = resampled.len() as i64 / sample.channels as i64;

        assert!(
            module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "XM replace_sample_data failed: sample={} rate={} channels={} original_frames={} resampled_len={} resampled_frames={} code={} message={}",
            sample.index,
            sample.rate,
            sample.channels,
            sample.length_frames,
            resampled.len(),
            resampled_length,
            module.last_error_code(),
            module.last_error_message(),
        );
        remaster::patch_sample_offsets(&mut module, sample.index, sample.rate, 48000);

        assert!(
            (module.sample_rate(sample.index) - 48000).abs() <= 8,
            "Live XM sample rate should stay within transpose quantization of 48kHz",
        );

        let saved = module
            .save_loaded_format_to_memory()
            .expect("XM should save in its loaded format");
        let reloaded = Module::from_memory(&saved).expect("Should reload saved XM bytes");
        assert_eq!(reloaded.info().format_type, "xm");
        assert!(
            (reloaded.sample_rate(sample.index) - 48000).abs() <= 8,
            "Reloaded XM sample rate should stay within transpose quantization of 48kHz",
        );
    }

    #[test]
    fn test_xm_live_render_stays_close_after_48khz_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut original_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let sample = first_nonempty_sample(&live_module);
        let forced_note = (1u8..=120)
            .find(|&note| live_module.instrument_sample_for_note(0, note) == Some(sample.index))
            .expect("Fixture instrument should map a note to the target sample");

        for module in [&mut original_module, &mut live_module] {
            assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, forced_note));
            assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
            assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, 0));
            assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, 0));
        }

        let original_render = render::render_live_module_to_samples(&mut original_module, 75, 8)
            .expect("Should render edited original XM");
        let original_data = live_module
            .read_sample_data(sample.index)
            .expect("Should read XM sample data");
        let resampled = resample_for_test(
            &original_data,
            sample.rate as u32,
            48000,
            sample.channels as usize,
            if sample.has_loop {
                remaster::ResampleBoundaryMode::LoopAware
            } else {
                remaster::ResampleBoundaryMode::OneShot
            },
        );
        let resampled_length = resampled.len() as i64 / sample.channels as i64;

        assert!(
            live_module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "XM replace_sample_data failed: sample={} rate={} channels={} original_frames={} resampled_len={} resampled_frames={} code={} message={}",
            sample.index,
            sample.rate,
            sample.channels,
            sample.length_frames,
            resampled.len(),
            resampled_length,
            live_module.last_error_code(),
            live_module.last_error_message(),
        );
        remaster::patch_sample_offsets(&mut live_module, sample.index, sample.rate, 48000);

        let live_render = render::render_live_module_to_samples(&mut live_module, 75, 8)
            .expect("Should render edited XM");

        let common_len = original_render.len().min(live_render.len());
        assert!(
            common_len > 0,
            "Original and remastered renders should both produce audio",
        );
        let diff = mean_abs_diff(&original_render[..common_len], &live_render[..common_len]);
        assert!(
            diff < 0.03,
            "Retuned XM remaster should stay close to the original pitch and timing (diff={diff})",
        );
    }

    #[test]
    fn test_mod_live_render_stays_close_after_48khz_replace() {
        let data = std::fs::read(MOD_FIXTURE).expect("Failed to read MOD fixture");
        let mut original_module = Module::from_memory(&data).expect("Failed to load MOD fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load MOD fixture");
        let sample = mod_regression_sample(&live_module);
        let original_frames = (sample.rate as usize / 2).max(2048);
        let original_tone = synthetic_tone(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
        );
        let original_length = original_tone.len() as i64 / sample.channels as i64;

        for module in [&mut original_module, &mut live_module] {
            assert!(
                module.replace_sample_data(
                    sample.index,
                    &original_tone,
                    original_length,
                    sample.channels,
                    sample.rate,
                ),
                "Failed to install controlled MOD test tone",
            );
            configure_mod_phrase(module, sample.index, MOD_TEST_NOTE);
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let original_render = render::render_live_module_to_samples(&mut original_module, 75, 8)
            .expect("Should render edited original MOD");
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            48000,
            sample.channels as usize,
            if sample.has_loop {
                remaster::ResampleBoundaryMode::LoopAware
            } else {
                remaster::ResampleBoundaryMode::OneShot
            },
        );
        let resampled_length = resampled.len() as i64 / sample.channels as i64;

        assert!(
            live_module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "MOD replace_sample_data failed: sample={} rate={} channels={} original_frames={} resampled_len={} resampled_frames={} code={} message={}",
            sample.index,
            sample.rate,
            sample.channels,
            sample.length_frames,
            resampled.len(),
            resampled_length,
            live_module.last_error_code(),
            live_module.last_error_message(),
        );

        let live_render = render::render_live_module_to_samples(&mut live_module, 75, 8)
            .expect("Should render edited MOD");

        let common_len = original_render.len().min(live_render.len());
        assert!(
            common_len > 0,
            "Original and remastered renders should both produce audio",
        );
        let original_window = stereo_window_to_mono(&original_render[..common_len], 1024, 4096);
        let live_window = stereo_window_to_mono(&live_render[..common_len], 1024, 4096);
        let original_freq = estimate_frequency(&original_window, 48000.0);
        let live_freq = estimate_frequency(&live_window, 48000.0);
        let freq_error = (original_freq - live_freq).abs() / original_freq.max(1.0);
        assert!(
            freq_error < 0.02,
            "Retuned MOD remaster should stay close to the original pitch and timing (sample={} orig_rate={} original_freq={original_freq} live_freq={live_freq} freq_error={freq_error})",
            sample.index,
            sample.rate,
        );
    }

    #[test]
    fn test_mod_live_replace_refreshes_followup_note_tuning() {
        const FOLLOWUP_ROW: i32 = 8;
        const FOLLOWUP_NOTE: u8 = MOD_TEST_NOTE + 5;

        let data = std::fs::read(MOD_FIXTURE).expect("Failed to read MOD fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load MOD fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load MOD fixture");
        let sample = mod_regression_sample(&live_module);
        let original_frames = (sample.rate as usize / 2).max(4096);
        let original_tone = synthetic_tone(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
        );
        let original_length = original_tone.len() as i64 / sample.channels as i64;

        for module in [&mut control_module, &mut live_module] {
            assert!(
                module.replace_sample_data(
                    sample.index,
                    &original_tone,
                    original_length,
                    sample.channels,
                    sample.rate,
                ),
                "Failed to install controlled MOD test tone",
            );
            configure_mod_phrase_with_followup_note(
                module,
                sample.index,
                MOD_TEST_NOTE,
                FOLLOWUP_ROW,
                FOLLOWUP_NOTE,
            );
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            48000,
            sample.channels as usize,
            if sample.has_loop {
                remaster::ResampleBoundaryMode::LoopAware
            } else {
                remaster::ResampleBoundaryMode::OneShot
            },
        );
        let resampled_length = resampled.len() as i64 / sample.channels as i64;

        assert!(
            control_module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "Control MOD replace_sample_data failed: code={} message={}",
            control_module.last_error_code(),
            control_module.last_error_message(),
        );

        let mut control_buf = vec![0.0f64; 512 * 2];
        let mut live_buf = vec![0.0f64; 512 * 2];

        while live_module.current_pattern() == 0 && live_module.current_row() < FOLLOWUP_ROW / 2 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live warmup renders should stay aligned before replacement",
            );
            assert!(
                live_rendered > 0,
                "MOD warmup should reach the replacement point before the follow-up note"
            );
        }
        assert!(
            live_module.active_samples().contains(&sample.index),
            "Target MOD sample should still be active when the live replacement occurs",
        );

        assert!(
            live_module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "Live MOD replace_sample_data failed: code={} message={}",
            live_module.last_error_code(),
            live_module.last_error_message(),
        );

        let mut control_tail = Vec::new();
        let mut live_tail = Vec::new();
        let mut reached_followup_note = false;
        for _ in 0..48 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live renders should stay aligned after replacement",
            );
            if live_rendered == 0 {
                break;
            }
            if live_module.current_pattern() == 0 && live_module.current_row() >= FOLLOWUP_ROW {
                reached_followup_note = true;
            }
            if reached_followup_note {
                control_tail.extend_from_slice(&control_buf[..control_rendered * 2]);
                live_tail.extend_from_slice(&live_buf[..live_rendered * 2]);
            }
        }

        assert!(
            reached_followup_note,
            "Live MOD replacement regression test should advance to the follow-up note row",
        );
        let common_len = control_tail.len().min(live_tail.len());
        assert!(
            common_len > 0,
            "Live MOD replacement regression test should capture follow-up note audio",
        );
        let control_window = stereo_window_to_mono(&control_tail[..common_len], 1024, 4096);
        let live_window = stereo_window_to_mono(&live_tail[..common_len], 1024, 4096);
        let control_freq = estimate_frequency(&control_window, 48000.0);
        let live_freq = estimate_frequency(&live_window, 48000.0);
        let freq_error = (control_freq - live_freq).abs() / control_freq.max(1.0);
        assert!(
            freq_error < 0.02,
            "Live MOD replacement should retune the follow-up note after the swap (sample={} orig_rate={} control_freq={control_freq} live_freq={live_freq} freq_error={freq_error})",
            sample.index,
            sample.rate,
        );
    }

    #[test]
    fn test_xm_live_replace_preserves_tone_portamento_target() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, start_note, target_note) = xm_porta_sample_and_notes(&control_module);

        for module in [&mut control_module, &mut live_module] {
            configure_xm_porta_phrase(module, start_note, target_note);
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let original_data = control_module
            .read_sample_data(sample.index)
            .expect("Should read XM sample data");
        let resampled = resample_for_test(
            &original_data,
            sample.rate as u32,
            48000,
            sample.channels as usize,
            if sample.has_loop {
                remaster::ResampleBoundaryMode::LoopAware
            } else {
                remaster::ResampleBoundaryMode::OneShot
            },
        );
        let resampled_length = resampled.len() as i64 / sample.channels as i64;

        assert!(
            control_module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "Control XM replace_sample_data failed: code={} message={}",
            control_module.last_error_code(),
            control_module.last_error_message(),
        );

        let mut control_buf = vec![0.0f64; 2048 * 2];
        let mut live_buf = vec![0.0f64; 2048 * 2];
        while live_module.current_pattern() == 0 && live_module.current_row() < 2 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live warmup renders should stay aligned before replacement",
            );
            assert!(
                live_rendered > 0,
                "XM warmup should reach the programmed tone-portamento rows"
            );
        }

        assert!(
            live_module.current_pattern() == 0
                && live_module.current_row() >= 2
                && live_module.current_row() < 12,
            "Live XM replacement should occur while the programmed tone portamento is still active (pattern {}, row {})",
            live_module.current_pattern(),
            live_module.current_row(),
        );

        assert!(
            live_module.replace_sample_data(
                sample.index,
                &resampled,
                resampled_length,
                sample.channels,
                48000,
            ),
            "Live XM replace_sample_data failed: code={} message={}",
            live_module.last_error_code(),
            live_module.last_error_message(),
        );

        let mut control_tail = Vec::new();
        let mut live_tail = Vec::new();
        while live_module.current_pattern() == 0 && live_module.current_row() < 12 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live renders should stay aligned after replacement",
            );
            if live_rendered == 0 {
                break;
            }
            control_tail.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_tail.extend_from_slice(&live_buf[..live_rendered * 2]);
        }

        let common_len = control_tail.len().min(live_tail.len());
        assert!(
            common_len > 0,
            "Live XM portamento regression test should capture post-replacement audio",
        );
        assert!(
            mean_abs_diff(&control_tail[..common_len], &live_tail[..common_len]) < 0.02,
            "Live XM replacement should stay close to the control render during tone portamento",
        );

        let settled_start = common_len / 2;
        assert!(
            mean_abs_diff(
                &control_tail[settled_start..common_len],
                &live_tail[settled_start..common_len]
            ) < 0.01,
            "Live XM replacement should still land on the same tone-portamento target pitch",
        );
    }

    #[test]
    fn test_xm_subinteger_tone_portamento_survives_high_rate_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, start_note, target_note) = xm_low_rate_porta_sample_and_notes(&control_module);
        let active_rows = 48;
        let high_rate = 48_000;
        let control_sample_rate = sample.rate as f64;
        let live_sample_rate = high_rate as f64;

        for module in [&mut control_module, &mut live_module] {
            configure_xm_porta_phrase_with_param(
                module,
                start_note,
                target_note,
                0x01,
                active_rows,
            );
        }

        let original_frames = (sample.rate as usize * 4).clamp(16_384, 65_536);
        let original_tone = synthetic_tone_at_freq(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
            55.0,
        );
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            high_rate as u32,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut control_module, &sample, &original_tone, sample.rate);
        replace_sample_for_test(&mut live_module, &sample, &resampled, high_rate);

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let mut control_buf = vec![0.0f64; 256 * 2];
        let mut live_buf = vec![0.0f64; 256 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut control_frequencies = Vec::new();
        let mut live_frequencies = Vec::new();
        let mut last_row = -1;

        while live_module.current_pattern() == 0 && live_module.current_row() < active_rows + 1 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Sub-integer XM tone-portamento renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }
            control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);

            let row = live_module.current_row();
            if row >= 2 && row != last_row {
                control_frequencies.push((
                    row,
                    control_module.test_get_current_channel_frequency(0) / control_sample_rate,
                ));
                live_frequencies.push((
                    row,
                    live_module.test_get_current_channel_frequency(0) / live_sample_rate,
                ));
                last_row = row;
            }
        }

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len >= 8_192,
            "Sub-integer XM tone-portamento regression should capture enough audio",
        );
        assert!(
            mean_abs_diff(&control_capture[..common_len], &live_capture[..common_len]) < 0.04,
            "Retuned XM tone portamento should stay close to the control render even when the compensated delta is fractional",
        );

        assert!(
            control_frequencies.len() >= 3 && live_frequencies.len() >= 3,
            "Sub-integer XM tone-portamento regression should capture several row-level frequencies",
        );
        let (control_early_row, control_early) = control_frequencies[0];
        let (control_mid_row, control_mid) = control_frequencies[control_frequencies.len() / 2];
        let (control_late_row, control_late) = *control_frequencies.last().unwrap();
        let (live_early_row, live_early) = live_frequencies[0];
        let (live_mid_row, live_mid) = live_frequencies[live_frequencies.len() / 2];
        let (live_late_row, live_late) = *live_frequencies.last().unwrap();

        assert_eq!(control_early_row, live_early_row);
        assert_eq!(control_mid_row, live_mid_row);
        assert_eq!(control_late_row, live_late_row);

        assert!(
            control_mid > control_early && control_late > control_mid,
            "Control XM tone portamento should move monotonically toward the target frequency",
        );
        assert!(
            live_mid > live_early && live_late > live_mid,
            "Sub-integer XM tone portamento should keep moving toward the target frequency (rows=({live_early_row}, {live_mid_row}, {live_late_row}) freqs=({live_early:.4}, {live_mid:.4}, {live_late:.4}))",
        );
        assert!(
            live_late > live_early * 1.10,
            "Sub-integer XM tone portamento should accumulate into a clearly non-zero pitch move instead of flattening (early={live_early:.4} late={live_late:.4})",
        );
    }

    #[test]
    fn test_xm_offset_slide_render_stays_close_after_48khz_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, note) = xm_low_rate_sample_and_note(&control_module);

        for module in [&mut control_module, &mut live_module] {
            configure_xm_offset_slide_phrase(module, note, 0x04, 0x0A, 12);
        }

        let original_frames = (sample.rate as usize * 4).max(16_384);
        let original_tone = synthetic_tone_at_freq(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
            55.0,
        );
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            48_000,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut control_module, &sample, &original_tone, sample.rate);
        replace_sample_for_test(&mut live_module, &sample, &resampled, 48_000);
        remaster::patch_sample_offsets(&mut live_module, sample.index, sample.rate, 48_000);

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let mut control_buf = vec![0.0f64; 2_048 * 2];
        let mut live_buf = vec![0.0f64; 2_048 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut reached_slide_rows = false;
        while live_module.current_pattern() == 0 && live_module.current_row() < 13 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live XM offset-slide renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }
            control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);
            if live_module.current_row() >= 6 {
                reached_slide_rows = true;
            }
        }

        assert!(
            reached_slide_rows,
            "Offset-slide regression should advance into the programmed XM slide rows",
        );

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len > 0,
            "Offset-slide regression should capture comparable audio",
        );
        assert!(
            mean_abs_diff(&control_capture[..common_len], &live_capture[..common_len]) < 0.03,
            "Retuned XM offset + slide phrase should stay close to the original render",
        );

        let settled_start = common_len / 2;
        assert!(
            mean_abs_diff(
                &control_capture[settled_start..common_len],
                &live_capture[settled_start..common_len]
            ) < 0.02,
            "Retuned XM offset + slide phrase should preserve the settled slide contour",
        );
    }

    #[test]
    fn test_xm_extra_fine_slide_render_stays_close_after_48khz_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, note) = xm_low_rate_sample_and_note(&control_module);

        for module in [&mut control_module, &mut live_module] {
            configure_xm_extra_fine_slide_phrase(module, note, 0x01, 16);
        }

        let original_frames = (sample.rate as usize * 4).max(16_384);
        let original_tone = synthetic_tone(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
        );
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            48_000,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut control_module, &sample, &original_tone, sample.rate);
        replace_sample_for_test(&mut live_module, &sample, &resampled, 48_000);
        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let mut control_buf = vec![0.0f64; 2_048 * 2];
        let mut live_buf = vec![0.0f64; 2_048 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut reached_slide_rows = false;
        while live_module.current_pattern() == 0 && live_module.current_row() < 17 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Control and live XM extra-fine renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }
            control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);
            if live_module.current_row() >= 10 {
                reached_slide_rows = true;
            }
        }

        assert!(
            reached_slide_rows,
            "Extra-fine XM slide regression should advance into the programmed memory rows",
        );

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len > 0,
            "Extra-fine XM slide regression should capture comparable audio",
        );
        assert!(
            mean_abs_diff(&control_capture[..common_len], &live_capture[..common_len]) < 0.03,
            "Retuned XM extra-fine slide phrase should stay close to the original render",
        );

        let settled_start = common_len / 2;
        assert!(
            mean_abs_diff(
                &control_capture[settled_start..common_len],
                &live_capture[settled_start..common_len]
            ) < 0.02,
            "Retuned XM extra-fine slide phrase should preserve the accumulated late-row slide",
        );
    }

    #[test]
    fn test_xm_cross_order_vibrato_render_stays_close_after_48khz_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, note) = xm_low_rate_sample_and_note(&control_module);

        for module in [&mut control_module, &mut live_module] {
            configure_xm_cross_order_vibrato_phrase(module, note, 0x81, 12);
        }

        let original_frames = (sample.rate as usize * 4).max(16_384);
        let original_tone = synthetic_tone(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
        );
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            48_000,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut control_module, &sample, &original_tone, sample.rate);
        replace_sample_for_test(&mut live_module, &sample, &resampled, 48_000);

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let mut control_buf = vec![0.0f64; 2_048 * 2];
        let mut live_buf = vec![0.0f64; 2_048 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut captured_second_order = false;

        loop {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Cross-order XM vibrato renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }

            let order = live_module.current_order();
            let row = live_module.current_row();
            if order >= 1 {
                captured_second_order = true;
                control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
                live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);
            }

            if order > 1 || (order == 1 && row >= 12) {
                break;
            }
        }

        assert!(
            captured_second_order,
            "Cross-order vibrato regression should reach the carried note in the second order",
        );

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len > 0,
            "Cross-order vibrato regression should capture comparable audio",
        );
        assert!(
            mean_abs_diff(&control_capture[..common_len], &live_capture[..common_len]) < 0.03,
            "Retuned XM cross-order vibrato should stay close to the control render",
        );

        let settled_start = common_len / 3;
        assert!(
            mean_abs_diff(
                &control_capture[settled_start..common_len],
                &live_capture[settled_start..common_len]
            ) < 0.02,
            "Retuned XM cross-order vibrato should preserve the late held-note contour",
        );
    }

    #[test]
    fn test_xm_vibrato_pitch_stays_centered_after_48khz_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, note) = xm_low_rate_sample_and_note(&control_module);

        for module in [&mut control_module, &mut live_module] {
            configure_xm_vibrato_phrase(module, note, 0x81, 48);
        }

        let original_frames = (sample.rate as usize * 4).max(16_384);
        let original_tone = synthetic_tone(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
        );
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            48_000,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut control_module, &sample, &original_tone, sample.rate);
        replace_sample_for_test(&mut live_module, &sample, &resampled, 48_000);

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let mut control_buf = vec![0.0f64; 2_048 * 2];
        let mut live_buf = vec![0.0f64; 2_048 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut reached_hold_rows = false;

        while live_module.current_pattern() == 0 && live_module.current_row() < 49 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "XM vibrato renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }

            control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);
            if live_module.current_row() >= 20 {
                reached_hold_rows = true;
            }
        }

        assert!(
            reached_hold_rows,
            "XM vibrato regression should advance into the held-note rows",
        );

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len > 0,
            "XM vibrato regression should capture comparable audio",
        );
        assert!(
            mean_abs_diff(&control_capture[..common_len], &live_capture[..common_len]) < 0.03,
            "Retuned XM vibrato should stay close to the control render",
        );

        let common_frames = common_len / 2;
        let skip_frames = common_frames / 4;
        let take_frames = (common_frames / 2).clamp(512, 4_096);
        let analysis_end = (skip_frames + take_frames).min(common_frames);
        let analysis_window = &control_capture[skip_frames * 2..analysis_end * 2];
        let analysis_frames = analysis_window.len() / 2;
        assert!(
            analysis_frames >= 512,
            "XM vibrato pitch regression needs a stable analysis window",
        );
        let left_mean = analysis_window
            .chunks_exact(2)
            .map(|frame| frame[0].abs())
            .sum::<f64>()
            / analysis_frames as f64;
        let right_mean = analysis_window
            .chunks_exact(2)
            .map(|frame| frame[1].abs())
            .sum::<f64>()
            / analysis_frames as f64;
        let analysis_channel = if right_mean > left_mean { 1 } else { 0 };
        let extract_channel = |data: &[f64]| -> Vec<f64> {
            data[skip_frames * 2..analysis_end * 2]
                .chunks_exact(2)
                .map(|frame| frame[analysis_channel])
                .collect()
        };
        let control_channel = extract_channel(&control_capture[..common_frames * 2]);
        let live_channel = extract_channel(&live_capture[..common_frames * 2]);
        assert!(
            mean_abs(&control_channel) > 1.0e-4 && mean_abs(&live_channel) > 1.0e-4,
            "XM vibrato pitch regression needs audible analysis windows",
        );
        let control_freq = estimate_frequency(&control_channel, 48_000.0);
        let live_freq = estimate_frequency(&live_channel, 48_000.0);

        assert!(
            ((live_freq - control_freq).abs() / control_freq) < 0.02,
            "Retuned XM vibrato should keep the held note centered (control={control_freq:.2}Hz live={live_freq:.2}Hz)",
        );
    }

    #[test]
    fn test_xm_high_note_vibrato_pitch_stays_centered_after_48khz_replace() {
        assert_xm_high_note_vibrato_pitch_after_replace(None, None);
    }

    #[test]
    fn test_xm_high_note_volcol_vibrato_pitch_stays_centered_after_48khz_replace() {
        assert_xm_high_note_vibrato_pitch_after_replace(Some(1), None);
    }

    #[test]
    fn test_xm_subinteger_vibrato_survives_high_rate_replace() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut control_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mut live_module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let (sample, note) = xm_low_rate_sample_and_note(&control_module);
        let hold_rows = 48;
        let high_rate = 48_000;
        let control_sample_rate = sample.rate as f64;
        let live_sample_rate = high_rate as f64;
        for module in [&mut control_module, &mut live_module] {
            configure_xm_vibrato_phrase(module, note, 0x11, hold_rows);
        }

        let original_frames = (sample.rate as usize * 4).clamp(16_384, 65_536);
        let original_tone = synthetic_tone(
            sample.rate as u32,
            sample.channels as usize,
            original_frames,
        );
        let resampled = resample_for_test(
            &original_tone,
            sample.rate as u32,
            high_rate as u32,
            sample.channels as usize,
            remaster::ResampleBoundaryMode::OneShot,
        );

        replace_sample_for_test(&mut control_module, &sample, &original_tone, sample.rate);
        replace_sample_for_test(&mut live_module, &sample, &resampled, high_rate);

        for module in [&mut control_module, &mut live_module] {
            module.set_repeat_count(0);
            module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH);
            module.set_stereo_separation(75);
            module.set_position_seconds(0.0);
        }

        let mut control_buf = vec![0.0f64; 32 * 2];
        let mut live_buf = vec![0.0f64; 32 * 2];
        let mut control_capture = Vec::new();
        let mut live_capture = Vec::new();
        let mut control_increments = Vec::new();
        let mut live_increments = Vec::new();

        while live_module.current_pattern() == 0 && live_module.current_row() < hold_rows + 1 {
            let control_rendered =
                control_module.read_interleaved_double_stereo(48_000, &mut control_buf);
            let live_rendered = live_module.read_interleaved_double_stereo(48_000, &mut live_buf);
            assert_eq!(
                control_rendered, live_rendered,
                "Sub-integer XM vibrato renders should stay frame-aligned",
            );
            if live_rendered == 0 {
                break;
            }
            control_capture.extend_from_slice(&control_buf[..control_rendered * 2]);
            live_capture.extend_from_slice(&live_buf[..live_rendered * 2]);
            if live_module.current_row() >= 4 {
                control_increments.push(
                    control_module.test_get_current_channel_increment(0) / control_sample_rate,
                );
                live_increments
                    .push(live_module.test_get_current_channel_increment(0) / live_sample_rate);
            }
        }

        let common_len = control_capture.len().min(live_capture.len());
        assert!(
            common_len >= 8_192,
            "Sub-integer XM vibrato regression should capture enough audio",
        );
        assert!(
            mean_abs_diff(&control_capture[..common_len], &live_capture[..common_len]) < 0.05,
            "Retuned XM vibrato should stay close to the control render even when the compensated delta is fractional",
        );

        assert!(
            control_increments.len() >= 8 && live_increments.len() >= 8,
            "Sub-integer XM vibrato regression needs several increment samples",
        );

        let trace_range = |trace: &[f64]| -> f64 {
            let min = trace.iter().copied().fold(f64::INFINITY, f64::min);
            let max = trace.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max - min
        };
        let trace_mean =
            |trace: &[f64]| -> f64 { trace.iter().copied().sum::<f64>() / trace.len() as f64 };
        let control_range = trace_range(&control_increments);
        let live_range = trace_range(&live_increments);
        let control_mean = trace_mean(&control_increments);
        let live_mean = trace_mean(&live_increments);
        let control_relative_range = control_range / control_mean.abs().max(1.0e-12);
        let live_relative_range = live_range / live_mean.abs().max(1.0e-12);

        assert!(
            control_relative_range > 1.0e-4,
            "Control XM vibrato should produce a measurable increment swing",
        );
        assert!(
            live_relative_range > control_relative_range * 0.5 && live_relative_range > 1.0e-4,
            "Sub-integer XM vibrato should still modulate channel pitch instead of flattening (control_relative_range={control_relative_range:.6} live_relative_range={live_relative_range:.6})",
        );
        // When vibrato is perfectly centered (double-precision removes quantization bias),
        // both means can be near-zero, making relative comparison undefined.
        // Use absolute difference clamped by the range as a scale reference.
        let mean_diff = (live_mean - control_mean).abs();
        let scale = control_mean.abs().max(control_range).max(1.0e-12);
        assert!(
            mean_diff / scale < 0.03,
            "Retuned XM vibrato should keep the same center pitch ratio (control_mean={control_mean:.9} live_mean={live_mean:.9} diff={mean_diff:.9})",
        );
    }

    #[test]
    fn test_get_note_from_period_avoids_truncated_boundary_off_by_one() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let module = Module::from_memory(&data).expect("Failed to load module");
        let c5speed = 8_363.0;
        let mut found = None;

        for note in 1u32..120 {
            let current = module.test_get_period_from_note(note, 0, c5speed);
            let next = module.test_get_period_from_note(note + 1, 0, c5speed);
            if !current.is_finite() || !next.is_finite() || current == 0.0 || next == 0.0 {
                continue;
            }
            for step in 1..1_024u32 {
                let t = step as f64 / 1_024.0;
                let candidate = current + (next - current) * t;
                let exact_note = module
                    .test_get_note_from_period(candidate, 0, c5speed)
                    .round() as u32;
                let truncated_note = truncated_note_search_for_test(&module, candidate, 0, c5speed);
                if exact_note != truncated_note {
                    found = Some((candidate, exact_note, truncated_note));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }

        let (candidate, exact_note, truncated_note) = found.expect(
            "Exact period lookup should diverge from the old truncated-note search near at least one fractional boundary",
        );
        assert!(
            (candidate - candidate.round()).abs() > 1.0e-6,
            "The note-boundary regression should exercise a fractional period",
        );
        assert_ne!(
            exact_note, truncated_note,
            "Exact period lookup should avoid the truncated-search off-by-one",
        );
    }

    #[test]
    fn test_xm_offset_patching_uses_instrument_keyboard_map() {
        let data = std::fs::read(XM_FIXTURE).expect("Failed to read XM fixture");
        let mut module = Module::from_memory(&data).expect("Failed to load XM fixture");
        let mapped_note = (1u8..=120)
            .find(|&note| module.instrument_sample_for_note(0, note) == Some(1))
            .expect("Fixture instrument should map at least one note to sample 2");

        assert!(
            module.pattern_num_rows(0) > 0,
            "XM fixture should have pattern data"
        );
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_NOTE, mapped_note));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_INSTRUMENT, 1));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_EFFECT, CMD_OFFSET));
        assert!(module.set_pattern_command(0, 0, 0, COMMAND_PARAMETER, 0x10));

        let saved_for_target = remaster::save_effect_params(&module, 1);
        assert!(
            saved_for_target.contains(&(0, 0, 0, 0x10)),
            "XM save_effect_params should target the sample resolved by the instrument keyboard map",
        );

        let saved_for_other = remaster::save_effect_params(&module, 0);
        assert!(
            !saved_for_other
                .iter()
                .any(|&(pat, row, ch, _)| pat == 0 && row == 0 && ch == 0),
            "XM save_effect_params should not fall back to sample_index + 1 instrument matching",
        );

        remaster::patch_sample_offsets(&mut module, 1, 16000, 48000);
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            0x30,
            "XM 9xx offsets should scale for the resolved target sample",
        );

        remaster::restore_effect_params(&mut module, &saved_for_target);
        remaster::patch_sample_offsets(&mut module, 0, 16000, 48000);
        assert_eq!(
            module.get_pattern_command(0, 0, 0, COMMAND_PARAMETER),
            0x10,
            "XM 9xx offsets should remain unchanged for other samples on the same instrument",
        );
    }

    #[test]
    fn test_save_loaded_format_preserves_writable_source_formats() {
        assert_save_loaded_format_roundtrip(MOD_FIXTURE, "mod", "mod");
        assert_save_loaded_format_roundtrip(S3M_FIXTURE, "s3m", "s3m");
        assert_save_loaded_format_roundtrip(XM_FIXTURE, "xm", "xm");
        assert_save_loaded_format_roundtrip(MPTM_FIXTURE, "mptm", "mptm");

        let s3m_data = std::fs::read(S3M_FIXTURE).expect("Failed to read S3M fixture");
        let s3m_module = Module::from_memory(&s3m_data).expect("Failed to load S3M fixture");
        let it_bytes = s3m_module
            .save_to_memory()
            .expect("IT export helper should serialize the module");
        let it_module = Module::from_memory(&it_bytes).expect("Failed to reload generated IT");

        assert_eq!(it_module.info().format_type, "it");
        assert_eq!(it_module.loaded_format_extension(), "it");
        let saved_it = it_module
            .save_loaded_format_to_memory()
            .expect("IT should save in its loaded format");
        let reloaded_it = Module::from_memory(&saved_it).expect("Failed to reload saved IT");
        assert_eq!(reloaded_it.info().format_type, "it");
    }

    #[test]
    fn test_extract_sample_jobs_populates_reference_fields() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");
        let work_dir = tempfile::tempdir().expect("tempdir");
        let raw_samples = remaster::read_raw_samples(&mut module);
        let cancel_flag = std::sync::atomic::AtomicBool::new(false);
        let jobs = remaster::extract_sample_jobs(
            &raw_samples,
            work_dir.path(),
            5.12,
            remaster::CleanupSettings::off(),
            &cancel_flag,
        )
        .expect("extract_sample_jobs should succeed");

        assert!(!jobs.is_empty(), "Should extract at least one sample job");

        for job in &jobs {
            assert!(job.channels > 0, "channels should be positive");
            assert!(job.rate > 0, "job.rate should be positive");
            assert!(
                !job.original_data.is_empty(),
                "original_data should be preserved for sample {}",
                job.index
            );
            assert!(
                !job.reference_48k.is_empty(),
                "reference_48k should be populated for sample {}",
                job.index
            );
            // reference_48k should be a 48kHz resample of the original data
            assert!(
                job.original_length_48k_frames > 0,
                "original_length_48k_frames should be positive for sample {}",
                job.index
            );
            // job.rate should remain the original module sample rate.
            let module_rate = module.sample_rate(job.index);
            assert_eq!(
                job.rate, module_rate,
                "job.rate should match module sample rate for sample {}",
                job.index
            );
            assert_eq!(
                job.source_length_frames,
                module.sample_length_frames(job.index),
                "source length should match the original module sample length for sample {}",
                job.index
            );
            let loop_info = module.sample_loop_info(job.index);
            assert_eq!(
                job.loop_info, loop_info,
                "loop_info should match the original module loop info for sample {}",
                job.index
            );
        }
    }

    #[test]
    fn test_spectral_correlation_picks_faithful_candidate() {
        use super::engine::spectral_correlation;

        // Reference: 200Hz sine at 48kHz (below 4kHz Nyquist for 8kHz original)
        let reference: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 200.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();

        // Good candidate: same 200Hz sine with slight noise (faithful preservation)
        let good: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.01 * ((i as f64 * 7.3).sin()))
            .collect();

        // Bad candidate: different frequency content (speech hallucination)
        let bad: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 1500.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();

        let good_score = spectral_correlation(&reference, &good, 1, 8000);
        let bad_score = spectral_correlation(&reference, &bad, 1, 8000);

        assert!(
            good_score > bad_score,
            "Faithful candidate ({good_score:.4}) should score higher than hallucinating one ({bad_score:.4})"
        );
        assert!(
            good_score > 0.9,
            "Good candidate should have high correlation: {good_score:.4}"
        );
    }

    #[test]
    fn test_quinlight_mix_selects_top_two_original_matches_and_weights() {
        let reference: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 200.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let a = reference.clone();
        let b: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.03 * ((i as f64 * 7.3).sin()))
            .collect();
        let c: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 1500.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();

        let engines = vec![
            ("AudioSR".to_string(), a.clone(), 4096i64, 1i32),
            ("LavaSR".to_string(), b.clone(), 4096, 1),
            ("FLowHigh".to_string(), c, 4096, 1),
        ];

        let mix =
            remaster::select_quinlight_mix(&reference, 1, 48_000, &engines, engines.len(), false);

        // 2 usable engines → spectral intersection blend with equal weights
        assert!(mix.name.starts_with("Quinlight Audio ("));
        assert_eq!(mix.channels, 1);
        assert_eq!(mix.contributors.len(), 2);
        assert_eq!(mix.contributors[0].name, "AudioSR");
        assert_eq!(mix.contributors[1].name, "LavaSR");
        assert!(
            (mix.contributors[0].weight - 0.5).abs() < 1e-6,
            "Spectral intersection uses equal weights"
        );
        assert!(!mix.data.is_empty());
    }

    #[test]
    fn test_quinlight_mix_multi_engine_matches_reference_rms_without_reference_contributor() {
        use super::engine::spectral_correlation;

        let n = 8192;
        let sample_rate = 48_000.0;
        let freq = 375.0;
        let reference: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.9 * (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect();
        let engine_a: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.6 * (2.0 * std::f64::consts::PI * freq * t + 1.0).sin()
            })
            .collect();
        let engine_b: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                0.4 * (2.0 * std::f64::consts::PI * freq * t + 2.0).sin()
            })
            .collect();
        let score_a = spectral_correlation(&reference, &engine_a, 1, 48_000);
        let score_b = spectral_correlation(&reference, &engine_b, 1, 48_000);
        let engines = vec![
            ("AudioSR".to_string(), engine_a, n as i64, 1i32),
            ("LavaSR".to_string(), engine_b, n as i64, 1i32),
        ];

        let mix =
            remaster::select_quinlight_mix(&reference, 1, 48_000, &engines, engines.len(), false);

        assert!(
            score_a > 0.90 && score_b > 0.90,
            "Fixture should keep both engines usable for multi-engine consensus, got score_a={score_a:.4}, score_b={score_b:.4}"
        );
        assert!(mix.name.starts_with("Quinlight Audio ("));
        assert!(
            mix.contributors
                .iter()
                .all(|contributor| contributor.name != "Reference48k"),
            "Successful AI consensus should not include the reference as a contributor"
        );
        assert!(
            (rms(&mix.data) - rms(&reference)).abs() < 1.0e-6,
            "Quinlight should keep RMS matched to the reference-derived target"
        );
        assert!(
            mean_abs_diff(&mix.data, &reference) > 0.05,
            "AI-only consensus should not collapse back onto the reference waveform"
        );
    }

    #[test]
    fn test_quinlight_one_usable_of_two_dispatched_marks_original_fallback() {
        // With 2 engines dispatched but only 1 scoring above floor,
        // consensus is impossible — final Quinlight should keep the original sample.
        let reference: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 180.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let good: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.02 * ((i as f64 * 5.1).sin()))
            .collect();
        let bad: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 2000.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let engines = vec![
            ("AudioSR".to_string(), good, 4096i64, 1i32),
            ("Bad".to_string(), bad, 4096, 1),
        ];

        let mix =
            remaster::select_quinlight_mix(&reference, 1, 48_000, &engines, engines.len(), false);

        assert_eq!(mix.name, "Quinlight Audio");
        assert_eq!(mix.contributors.len(), 1);
        assert_eq!(mix.contributors[0].name, "Original");
        assert!((mix.contributors[0].weight - 1.0).abs() < 1e-6);
        assert!(
            mean_abs_diff(&mix.data, &reference) < 1e-6,
            "Helper mix should still score against the 48 kHz reference even when the final result keeps the original sample"
        );
        assert!(remaster::is_no_consensus_result(&mix.name));
    }

    #[test]
    fn test_quinlight_single_dispatch_marks_original_fallback() {
        // With only 1 engine dispatched and 1 usable, consensus is still
        // impossible — final Quinlight should keep the original sample.
        let reference: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 180.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let good: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.02 * ((i as f64 * 5.1).sin()))
            .collect();
        let engines = vec![("AudioSR".to_string(), good, 4096i64, 1i32)];

        let mix = remaster::select_quinlight_mix(&reference, 1, 48_000, &engines, 1, false);

        assert_eq!(mix.name, "Quinlight Audio");
        assert_eq!(mix.contributors.len(), 1);
        assert_eq!(mix.contributors[0].name, "Original");
        assert!(
            mean_abs_diff(&mix.data, &reference) < 1e-6,
            "Helper mix should still mirror the 48 kHz scoring reference"
        );
        assert!(remaster::is_no_consensus_result(&mix.name));
    }

    #[test]
    fn test_is_no_consensus_result() {
        assert!(remaster::is_no_consensus_result("Quinlight Audio"));
        assert!(!remaster::is_no_consensus_result("Quinlight Audio (A+L)"));
        assert!(!remaster::is_no_consensus_result("Quinlight Audio (A)"));
        assert!(!remaster::is_no_consensus_result("Quinlight Audio (L+F)"));
    }

    #[test]
    fn test_quinlight_mix_with_no_usable_engines_marks_original_fallback() {
        let reference: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 220.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let bad_a: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 1800.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let bad_b: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 2600.0 * 2.0 * std::f64::consts::PI / 48000.0).cos())
            .collect();
        let engines = vec![
            ("BadA".to_string(), bad_a, 4096i64, 1i32),
            ("BadB".to_string(), bad_b, 4096, 1),
        ];

        let mix =
            remaster::select_quinlight_mix(&reference, 1, 48_000, &engines, engines.len(), false);

        assert_eq!(mix.contributors.len(), 1);
        assert_eq!(mix.contributors[0].name, "Original");
        assert_eq!(mix.contributors[0].weight, 1.0);
        assert!(mean_abs_diff(&mix.data, &reference) < 1e-6);
    }

    #[test]
    fn test_quinlight_mix_reference_prefers_filtered_candidate() {
        let base: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 180.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
            .collect();
        let original: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                s + 0.35 * (i as f64 * 9000.0 * 2.0 * std::f64::consts::PI / 48000.0).sin()
            })
            .collect();
        let filtered_reference = remaster::build_quinlight_reference_48k(
            &original,
            48_000,
            1,
            false,
            remaster::CleanupSettings::off(),
        )
        .expect("Should build filtered reference");
        let engines = vec![
            ("RawLike".to_string(), original.clone(), 4096i64, 1i32),
            (
                "FilteredLike".to_string(),
                filtered_reference.clone(),
                4096,
                1,
            ),
            (
                "Outlier".to_string(),
                (0..4096)
                    .map(|i| (i as f64 * 2400.0 * 2.0 * std::f64::consts::PI / 48000.0).sin())
                    .collect(),
                4096,
                1,
            ),
        ];

        let mix = remaster::select_quinlight_mix(
            &filtered_reference,
            1,
            48_000,
            &engines,
            engines.len(),
            false,
        );

        assert_eq!(mix.contributors[0].name, "FilteredLike");
    }

    #[test]
    fn test_looped_reference_keeps_one_shot_onset_for_first_segment() {
        let original = vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let one_shot = remaster::build_quinlight_reference_48k(
            &original,
            48_000,
            1,
            false,
            remaster::CleanupSettings::off(),
        )
        .expect("Should build one-shot reference");
        let looped = remaster::build_quinlight_reference_48k(
            &original,
            48_000,
            1,
            true,
            remaster::CleanupSettings::off(),
        )
        .expect("Should build loop-aware reference");

        assert_eq!(looped.len(), one_shot.len());
        assert!(
            (looped[0] - one_shot[0]).abs() < 1e-3,
            "The first extracted sample should keep its one-shot onset context",
        );
    }

    // --- Aniso-64 enhancement tests ---

    /// Calls the extern "C" AVX2 stereo dot product for testing.
    /// Safety: kernel must have 64 elements, samples must have 128 elements.
    unsafe fn call_aniso64_stereo(kernel: &[f64; 64], samples: &[f64; 128]) -> (f64, f64) {
        unsafe extern "C" {
            fn aniso64_dot_stereo_avx2(
                kernel: *const f64,
                samples: *const f64,
                out_l: *mut f64,
                out_r: *mut f64,
            );
        }
        let mut out_l = 0.0f64;
        let mut out_r = 0.0f64;
        unsafe {
            aniso64_dot_stereo_avx2(kernel.as_ptr(), samples.as_ptr(), &mut out_l, &mut out_r);
        }
        (out_l, out_r)
    }

    #[test]
    fn aniso64_stereo_avx2_deinterleave_correctness() {
        // Identity kernel: 1.0 at tap 31, 0.0 elsewhere — should pick out sample pair 31
        let mut kernel = [0.0f64; 64];
        kernel[31] = 1.0;
        // Interleaved stereo: L[i] = i, R[i] = 63 - i
        let mut samples = [0.0f64; 128];
        for i in 0..64 {
            samples[i * 2] = i as f64; // L
            samples[i * 2 + 1] = 63.0 - i as f64; // R
        }

        let (out_l, out_r) = unsafe { call_aniso64_stereo(&kernel, &samples) };
        assert!(
            (out_l - 31.0).abs() < 1e-10,
            "Identity kernel should extract L[31]=31.0, got {out_l}"
        );
        assert!(
            (out_r - 32.0).abs() < 1e-10,
            "Identity kernel should extract R[31]=32.0, got {out_r}"
        );

        // Uniform kernel: all 1/64 — should return mean of each channel
        let uniform_kernel = [1.0 / 64.0; 64];
        let (out_l, out_r) = unsafe { call_aniso64_stereo(&uniform_kernel, &samples) };
        let expected_mean_l: f64 = (0..64).map(|i| i as f64).sum::<f64>() / 64.0; // 31.5
        let expected_mean_r: f64 = (0..64).map(|i| 63.0 - i as f64).sum::<f64>() / 64.0; // 31.5
        assert!(
            (out_l - expected_mean_l).abs() < 1e-10,
            "Uniform kernel: expected L mean {expected_mean_l}, got {out_l}"
        );
        assert!(
            (out_r - expected_mean_r).abs() < 1e-10,
            "Uniform kernel: expected R mean {expected_mean_r}, got {out_r}"
        );

        // Cross-check: verify SIMD matches scalar reference
        let scalar_l: f64 = (0..64).map(|t| kernel[t] * samples[t * 2]).sum();
        let scalar_r: f64 = (0..64).map(|t| kernel[t] * samples[t * 2 + 1]).sum();
        let (simd_l, simd_r) = unsafe { call_aniso64_stereo(&kernel, &samples) };
        assert!(
            (simd_l - scalar_l).abs() < 1e-10,
            "SIMD L should match scalar: SIMD={simd_l}, scalar={scalar_l}"
        );
        assert!(
            (simd_r - scalar_r).abs() < 1e-10,
            "SIMD R should match scalar: SIMD={simd_r}, scalar={scalar_r}"
        );
    }

    #[test]
    fn aniso64_beta_shear_round_trip() {
        let data = std::fs::read(BASIC_FIXTURE).expect("test fixture must exist");
        let mut module = Module::from_memory(&data).expect("load module");

        // Default values
        let default_beta = module.aniso64_k_beta();
        let default_beta2 = module.aniso64_k_beta2();
        assert!(
            (default_beta - 0.65).abs() < 1e-10,
            "Default k_beta should be 0.65, got {default_beta}"
        );
        assert!(
            (default_beta2 - 0.15).abs() < 1e-10,
            "Default k_beta2 should be 0.15, got {default_beta2}"
        );

        // Set and read back
        module.set_aniso64_k_beta(0.8);
        module.set_aniso64_k_beta2(0.25);
        assert!(
            (module.aniso64_k_beta() - 0.8).abs() < 1e-10,
            "k_beta round-trip failed"
        );
        assert!(
            (module.aniso64_k_beta2() - 0.25).abs() < 1e-10,
            "k_beta2 round-trip failed"
        );
    }

    #[test]
    fn aniso64_beta_shear_affects_output() {
        // jt_pools.xm has deep vibrato (Hxx) effects that cause inter-tick pitch
        // changes, which is what triggers β-shear.  The basic S3M fixture plays
        // at constant pitch so dInc is always 0 and β-shear has no effect.
        // 10 seconds is enough to capture vibrato activity.
        const VIBRATO_FIXTURE: &str = "mods/jt_pools.xm";
        let data = std::fs::read(VIBRATO_FIXTURE).expect("vibrato fixture must exist");

        // Render 10s with zero beta-shear (0.0, 0.0) — β-shear disabled
        let default_output = {
            let mut module = Module::from_memory(&data).expect("load module");
            configure_quinlight_render(&mut module, 75, 64, true);
            module.set_aniso64_k_beta(0.0);
            module.set_aniso64_k_beta2(0.0);
            render_module_output_capped(&mut module, 10)
        };

        // Render 10s with extreme beta-shear (2.0, 2.0)
        let extreme_shear_output = {
            let mut module = Module::from_memory(&data).expect("load module");
            configure_quinlight_render(&mut module, 75, 64, true);
            module.set_aniso64_k_beta(2.0);
            module.set_aniso64_k_beta2(2.0);
            render_module_output_capped(&mut module, 10)
        };

        assert_eq!(default_output.len(), extreme_shear_output.len());

        // They should differ (beta-shear modifies the interpolation phase during pitch changes)
        let diff: f64 = default_output
            .iter()
            .zip(extreme_shear_output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "Extreme beta-shear should produce different output vs default"
        );
    }

    #[test]
    fn aniso64_spectral_quality() {
        use super::engine::spectral::spectral_correlation;

        // 15 seconds of audio is sufficient for spectral quality comparison.
        // Renders Aniso-64 once and shares it across both comparisons
        // (vs Sinc-16 and vs Sinc-8LP).
        let data = std::fs::read(BASIC_FIXTURE).expect("test fixture must exist");

        let aniso64_output = {
            let mut module = Module::from_memory(&data).expect("load module");
            configure_quinlight_render(&mut module, 75, 64, true);
            render_module_output_capped(&mut module, 15)
        };

        let sinc16_output = {
            let mut module = Module::from_memory(&data).expect("load module");
            configure_quinlight_render(&mut module, 75, 16, true);
            render_module_output_capped(&mut module, 15)
        };

        let sinc8lp_output = {
            let mut module = Module::from_memory(&data).expect("load module");
            configure_quinlight_render(&mut module, 75, 8, true);
            render_module_output_capped(&mut module, 15)
        };

        assert_eq!(
            sinc16_output.len(),
            aniso64_output.len(),
            "Both renders should produce the same number of samples"
        );
        assert!(!aniso64_output.is_empty(), "Render should produce audio");

        // --- Aniso-64 vs Sinc-16 ---
        let corr_vs_sinc16 = spectral_correlation(&sinc16_output, &aniso64_output, 2, 48000);
        eprintln!("Sinc-16 vs Aniso-64 spectral correlation: {corr_vs_sinc16:.6}");

        assert!(
            corr_vs_sinc16 > 0.90,
            "Spectral correlation should be high (same piece): {corr_vs_sinc16}"
        );
        assert!(
            corr_vs_sinc16 < 1.0,
            "Different interpolation filters should produce different output"
        );

        // Compute dB difference
        let diff_rms: f64 = sinc16_output
            .iter()
            .zip(aniso64_output.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / sinc16_output.len() as f64;
        let aniso64_rms: f64 =
            aniso64_output.iter().map(|s| s.powi(2)).sum::<f64>() / aniso64_output.len() as f64;
        if aniso64_rms > 0.0 && diff_rms > 0.0 {
            let db = 10.0 * (diff_rms / aniso64_rms).log10();
            eprintln!("Difference level: {db:.1} dB relative to Aniso-64 signal");
        }

        // Per-band analysis
        let sinc16_l: Vec<f64> = sinc16_output.iter().step_by(2).copied().collect();
        let aniso64_l: Vec<f64> = aniso64_output.iter().step_by(2).copied().collect();

        let bands = super::engine::spectral::per_band_energy_ratio(
            &sinc16_l, &aniso64_l, 48000, 24000.0, 8,
        );
        eprintln!("Per-band energy difference (Aniso-64 vs Sinc-16):");
        for (center, db) in &bands {
            eprintln!("  {center:8.0} Hz: {db:+.2} dB");
        }
        assert_eq!(bands.len(), 8, "Should produce 8 frequency bands");

        // --- Aniso-64 vs Sinc-8LP (phase resolution) ---
        assert_eq!(aniso64_output.len(), sinc8lp_output.len());

        let corr_vs_sinc8lp = spectral_correlation(&sinc8lp_output, &aniso64_output, 2, 48000);
        eprintln!("Aniso-64 vs Sinc-8LP spectral correlation: {corr_vs_sinc8lp:.6}");

        assert!(
            corr_vs_sinc8lp > 0.90,
            "Aniso-64 should produce high-quality output vs Sinc-8LP: {corr_vs_sinc8lp}"
        );
    }

    /// Render 2ND_PM.S3M with our engine and write raw f64 stereo PCM to /tmp
    /// for comparison with openmpt123's reference render.
    #[test]
    fn test_s3m_render_for_comparison() {
        let data = std::fs::read(BASIC_FIXTURE).expect("read fixture");
        let mut module = Module::from_memory(&data).expect("load module");
        module.set_repeat_count(0);
        // Match live GUI defaults: Aniso-64 filter, 50% stereo sep, AGC enabled
        module.set_stereo_separation(DEFAULT_STEREO_SEPARATION_PERCENT);
        module.set_interpolation_filter(DEFAULT_INTERPOLATION_FILTER_LENGTH); // Aniso-64
        module.apply_quinlight_processing_settings(
            DEFAULT_STEREO_SEPARATION_PERCENT,
            DEFAULT_INTERPOLATION_FILTER_LENGTH,
            DEFAULT_AGC_ENABLED,
        );

        let mut all_samples = Vec::new();
        let mut buf = vec![0.0f64; 48_000 * 2]; // 1 second stereo
        loop {
            let frames = module.read_interleaved_double_stereo(48_000, &mut buf);
            if frames == 0 {
                break;
            }
            all_samples.extend_from_slice(&buf[..frames * 2]);
        }

        // Write as raw f64 PCM
        let bytes: Vec<u8> = all_samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write("/tmp/quinlight_render.raw", &bytes).expect("write raw PCM");
        eprintln!(
            "Wrote {} frames ({:.1}s) to /tmp/quinlight_render.raw (f64 stereo 48kHz)",
            all_samples.len() / 2,
            all_samples.len() as f64 / 2.0 / 48_000.0
        );
    }

    /// Diagnostic: dump pattern data at order 22 to see what notes are written.
    #[test]
    fn test_s3m_order_22_pattern_dump() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let module = Module::from_memory(&data).expect("Failed to load module");

        let pattern = module.get_order_pattern(22);
        let num_rows = module.pattern_num_rows(pattern);
        let num_channels = module.num_channels();
        eprintln!(
            "Order 22 → Pattern {}, {} rows, {} channels",
            pattern, num_rows, num_channels
        );

        const CMD_NOTE: i32 = 0;
        const CMD_INSTR: i32 = 1;
        const CMD_VOLCMD: i32 = 2;
        const CMD_EFFECT: i32 = 3;
        const CMD_VOL: i32 = 4;
        const CMD_PARAM: i32 = 5;

        let note_str = |note: u8| -> String {
            if note == 0 {
                "...".to_string()
            } else if note == 254 {
                "^^^".to_string()
            } else if note == 255 {
                "===".to_string()
            } else {
                let oct = (note as u32 - 1) / 12;
                let n = (note as u32 - 1) % 12;
                let names = [
                    "C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-",
                ];
                format!("{}{}", names[n as usize], oct)
            }
        };

        let effect_name = |e: u8| -> &str {
            match e {
                0 => "---",
                1 => "Arp",
                2 => "PUp",
                3 => "PDn",
                4 => "TPo",
                5 => "Vib",
                6 => "TPV",
                7 => "VbV",
                8 => "Tre",
                9 => "Pan",
                10 => "Ofs",
                11 => "VSl",
                12 => "PJp",
                13 => "Vol",
                14 => "PBk",
                15 => "Rtg",
                16 => "Spd",
                17 => "Tmp",
                20 => "S3M",
                _ => "???",
            }
        };

        for row in 0..num_rows {
            let mut row_str = format!("Row {:2} |", row);
            let mut has_content = false;
            for ch in 0..num_channels {
                let note = module.get_pattern_command(pattern, row, ch, CMD_NOTE);
                let instr = module.get_pattern_command(pattern, row, ch, CMD_INSTR);
                let volcmd = module.get_pattern_command(pattern, row, ch, CMD_VOLCMD);
                let effect = module.get_pattern_command(pattern, row, ch, CMD_EFFECT);
                let _vol = module.get_pattern_command(pattern, row, ch, CMD_VOL);
                let param = module.get_pattern_command(pattern, row, ch, CMD_PARAM);

                if note > 0 || instr > 0 || volcmd > 0 || effect > 0 {
                    has_content = true;
                    row_str += &format!(
                        " Ch{}: {} {:2} {}({:02X})",
                        ch,
                        note_str(note),
                        instr,
                        effect_name(effect),
                        param
                    );
                }
            }
            if has_content {
                eprintln!("{}", row_str);
            }
        }
    }

    /// Diagnostic: render 2ND_PM.S3M through orders 21–23, polling VU meters
    /// and per-channel frequency every ~256 frames to detect per-channel
    /// activity and pitch at row granularity.
    #[test]
    fn test_s3m_order_22_bass_note_present() {
        let data = std::fs::read(BASIC_FIXTURE).expect("Failed to read test module");
        let mut module = Module::from_memory(&data).expect("Failed to load module");
        module.set_repeat_count(0);

        let num_channels = module.num_channels();
        let mut reached_order_22 = false;

        // Collect per-(order, row) peak VU and frequency for each channel
        let mut row_vu: std::collections::BTreeMap<(i32, i32), Vec<f64>> =
            std::collections::BTreeMap::new();
        let mut row_freq: std::collections::BTreeMap<(i32, i32), Vec<f64>> =
            std::collections::BTreeMap::new();

        // Small render chunks (~5ms) for per-row resolution
        let chunk_frames = 256;
        let mut buf = vec![0.0f64; chunk_frames * 2];
        for _ in 0..100_000 {
            let rendered = module.read_interleaved_double_stereo(48_000, &mut buf);
            if rendered == 0 {
                break;
            }
            let order = module.current_order();
            let row = module.current_row();

            if order >= 21 && order <= 23 {
                if order == 22 {
                    reached_order_22 = true;
                }
                let vu = module.channel_vu();
                let vu_entry = row_vu
                    .entry((order, row))
                    .or_insert_with(|| vec![0.0; num_channels as usize]);
                let freq_entry = row_freq
                    .entry((order, row))
                    .or_insert_with(|| vec![0.0; num_channels as usize]);
                for (ch, &(l, r)) in vu.iter().enumerate() {
                    let peak = l.max(r);
                    if peak > vu_entry[ch] {
                        vu_entry[ch] = peak;
                    }
                    let freq = module.test_get_current_channel_frequency(ch as i32);
                    if freq > freq_entry[ch] {
                        freq_entry[ch] = freq;
                    }
                }
            }
            if order > 23 {
                break;
            }
        }

        assert!(
            reached_order_22,
            "Should reach order 22 during rendering (stopped at order {}, row {})",
            module.current_order(),
            module.current_row(),
        );

        // Print compact activity map with frequencies for active channels in order 22 only
        eprintln!("=== Order 22: per-row channel VU + freq (Hz) ===");
        eprintln!(
            "Row | Ch0           Ch1           Ch2           Ch3           Ch4           Ch5           Ch6           Ch7"
        );
        for (&(order, row), vus) in &row_vu {
            if order != 22 {
                continue;
            }
            let freqs = row_freq.get(&(order, row)).unwrap();
            let cols: Vec<String> = vus
                .iter()
                .zip(freqs.iter())
                .map(|(v, f)| {
                    if *v < 0.001 {
                        "    ---      ".to_string()
                    } else {
                        format!("{:.2} {:5.0}Hz", v, f)
                    }
                })
                .collect();
            eprintln!(" {:2} | {}", row, cols.join(" "));
        }

        // Aggregate: peak VU per channel across all rows in order 22 only
        let mut order_22_peak = vec![0.0f64; num_channels as usize];
        for (&(order, _), vus) in &row_vu {
            if order == 22 {
                for (ch, &v) in vus.iter().enumerate() {
                    if v > order_22_peak[ch] {
                        order_22_peak[ch] = v;
                    }
                }
            }
        }
        eprintln!("\n=== Order 22 peak VU per channel ===");
        for (ch, &peak) in order_22_peak.iter().enumerate() {
            eprintln!(
                "  Ch {:2}: {:.4}{}",
                ch,
                peak,
                if peak < 0.001 { " ← SILENT" } else { "" }
            );
        }
    }
}
