// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

//! HRTF binaural spatialization for headphone listening.
//!
//! Splits the stereo mix into L/R mono signals, positions them as virtual
//! speakers at ±30° azimuth via SOFA HRIR convolution (sofar crate), and sums
//! the binaural outputs. Gives an externalized "speakers in front of you"
//! soundstage instead of the typical in-head headphone panning.
//!
//! Uses the D1 HRIR dataset (44.1k/48k/96k SOFA files with 256–512 tap FIRs).
//! The native-rate file is selected automatically based on the audio device
//! sample rate, then only the selected FIR taps are resampled locally when
//! the device rate falls between the packaged SOFA rates.

use sofar::reader::{Filter, OpenOptions};
use sofar::render::Renderer;

/// Partition length for uniformly partitioned convolution.
/// 256 samples ≈ 2.7ms at 96kHz, 5.3ms at 48kHz.
const PARTITION_LEN: usize = 256;

/// Maximum frames fed to the HRTF processor per `process()` call. The audio
/// callback loops in `player::CALLBACK_CHUNK_FRAMES` slices, so this is a safe
/// ceiling (≥ the chunk size); inputs larger than this are clamped (see
/// `process`).
const MAX_CALLBACK_FRAMES: usize = 16384;

/// Gain compensation for two-source binaural summation.
/// Correlated signals (mono bass) sum to 2×; 0.5 restores unity.
const BINAURAL_SUM_GAIN: f64 = 0.5;

/// Left virtual speaker at +30° azimuth (30° to the left).
/// SOFA Cartesian: x = front, y = left, z = up.
const LEFT_SPEAKER: (f32, f32, f32) = (0.866, 0.5, 0.0);

/// Right virtual speaker at −30° azimuth (30° to the right).
const RIGHT_SPEAKER: (f32, f32, f32) = (0.866, -0.5, 0.0);

pub struct HrtfProcessor {
    left_renderer: Renderer,
    right_renderer: Renderer,
    partition_len: usize,

    // Input accumulation (mono f32, one per channel)
    in_left: Vec<f32>,
    in_right: Vec<f32>,
    in_len: usize,

    // Processing buffers (f32, partition_len each)
    proc_ll: Vec<f32>,
    proc_lr: Vec<f32>,
    proc_rl: Vec<f32>,
    proc_rr: Vec<f32>,

    // Output accumulation (interleaved f64 stereo)
    out_buf: Vec<f64>,
    out_len: usize,

    // Safety peak limiter envelope: instant attack, slow (~-12 dB/s) release
    // so one loud transient doesn't attenuate the rest of the session.
    peak_env: f64,
    /// Per-sample release factor derived from the sample rate.
    limiter_release: f64,
    /// Total interleaved samples of startup latency injected as zero-fill
    /// (non-partition-multiple callback sizes only). Constant after the
    /// startup transient; the player delays its dry path by this much so the
    /// wet/dry mix stays phase-aligned.
    latency_samples: usize,
    /// Latched when the renderer reports an error: `process` becomes a dry
    /// passthrough (never desyncs L/R or grows the input unbounded).
    failed: bool,
}

/// Apply HRTF binaural processing to a complete buffer of interleaved f64 stereo
/// samples (offline / non-real-time). Pads to a full partition boundary so all
/// input is processed, then trims back to the original length.
pub fn process_offline(samples: &mut Vec<f64>, sample_rate: u32) -> Result<(), String> {
    let mut processor = HrtfProcessor::try_new(sample_rate)?;
    let block_samples = processor.partition_len * 2;

    let original_len = samples.len();
    let padded_len = original_len.div_ceil(block_samples) * block_samples;
    samples.resize(padded_len, 0.0);

    // Process in chunks that fit the input buffer (MAX_CALLBACK_FRAMES stereo frames).
    let chunk_samples = MAX_CALLBACK_FRAMES * 2;
    for chunk in samples.chunks_mut(chunk_samples) {
        processor.process(chunk);
    }

    samples.truncate(original_len);
    Ok(())
}

/// Apply HRTF binaural processing with wet/dry mix (0=dry, 100=full wet).
/// For offline / non-real-time use.
pub fn process_offline_with_mix(
    samples: &mut Vec<f64>,
    sample_rate: u32,
    mix_percent: i32,
) -> Result<(), String> {
    if mix_percent >= 100 {
        return process_offline(samples, sample_rate);
    }
    if mix_percent <= 0 {
        return Ok(());
    }

    let dry = samples.clone();
    process_offline(samples, sample_rate)?;

    let wet_gain = mix_percent as f64 / 100.0;
    let dry_gain = 1.0 - wet_gain;
    for (wet, d) in samples.iter_mut().zip(dry.iter()) {
        *wet = *d * dry_gain + *wet * wet_gain;
    }
    Ok(())
}

impl HrtfProcessor {
    pub fn try_new(sample_rate: u32) -> Result<Self, String> {
        let (left_filter, right_filter, filter_len) = load_filters(sample_rate)?;

        let partition_len = PARTITION_LEN;

        let mut left_renderer = Renderer::builder(filter_len)
            .with_sample_rate(sample_rate as f32)
            .with_partition_len(partition_len)
            .with_left_delay(left_filter.ldelay)
            .with_right_delay(left_filter.rdelay)
            .build()
            .map_err(|e| format!("HRTF renderer (L): {e}"))?;
        left_renderer
            .set_filter(&left_filter)
            .map_err(|e| format!("HRTF filter (L): {e}"))?;

        let mut right_renderer = Renderer::builder(filter_len)
            .with_sample_rate(sample_rate as f32)
            .with_partition_len(partition_len)
            .with_left_delay(right_filter.ldelay)
            .with_right_delay(right_filter.rdelay)
            .build()
            .map_err(|e| format!("HRTF renderer (R): {e}"))?;
        right_renderer
            .set_filter(&right_filter)
            .map_err(|e| format!("HRTF filter (R): {e}"))?;

        let max_in = MAX_CALLBACK_FRAMES + partition_len;
        let max_out = (MAX_CALLBACK_FRAMES + partition_len) * 2;

        Ok(Self {
            left_renderer,
            right_renderer,
            partition_len,
            in_left: vec![0.0; max_in],
            in_right: vec![0.0; max_in],
            in_len: 0,
            proc_ll: vec![0.0; partition_len],
            proc_lr: vec![0.0; partition_len],
            proc_rl: vec![0.0; partition_len],
            proc_rr: vec![0.0; partition_len],
            out_buf: vec![0.0; max_out],
            out_len: 0,
            peak_env: 0.0,
            // -12 dB/s: 10^(-12 / (20 * rate)) per sample.
            limiter_release: 10f64.powf(-12.0 / (20.0 * sample_rate.max(1) as f64)),
            latency_samples: 0,
            failed: false,
        })
    }

    /// Interleaved samples of wet-path startup latency (see `latency_samples`).
    pub fn latency_samples(&self) -> usize {
        self.latency_samples
    }

    /// True once a renderer error latched the processor into dry passthrough.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Process interleaved f64 stereo audio in-place.
    /// Bridges between arbitrary callback buffer sizes and the fixed partition size.
    pub fn process(&mut self, data: &mut [f64]) {
        let plen = self.partition_len;
        if self.failed {
            return; // dry passthrough — see `failed`
        }

        // Deinterleave f64 stereo → mono f32 channels, accumulate. The input
        // buffers are sized for MAX_CALLBACK_FRAMES + one partition; clamp so
        // an oversized caller buffer can't index out of bounds.
        let frames = (data.len() / 2).min(self.in_left.len().saturating_sub(self.in_len));
        for i in 0..frames {
            self.in_left[self.in_len + i] = data[i * 2] as f32;
            self.in_right[self.in_len + i] = data[i * 2 + 1] as f32;
        }
        self.in_len += frames;

        // Process complete partition blocks
        while self.in_len >= plen {
            // Left virtual speaker: left channel → binaural L/R
            // A renderer error latches dry passthrough: continuing after a
            // one-sided failure would advance one renderer's overlap state
            // and permanently skew L vs R, and the unconsumed input would
            // grow until it indexed out of bounds inside an audio callback.
            if self
                .left_renderer
                .process_block(
                    &self.in_left[..plen],
                    &mut self.proc_ll[..],
                    &mut self.proc_lr[..],
                )
                .is_err()
            {
                self.failed = true;
                return;
            }

            // Right virtual speaker: right channel → binaural L/R
            if self
                .right_renderer
                .process_block(
                    &self.in_right[..plen],
                    &mut self.proc_rl[..],
                    &mut self.proc_rr[..],
                )
                .is_err()
            {
                self.failed = true;
                return;
            }

            // Sum, interleave, and peak-limit output
            let start = self.out_len;
            for i in 0..plen {
                let l = (self.proc_ll[i] + self.proc_rl[i]) as f64 * BINAURAL_SUM_GAIN;
                let r = (self.proc_lr[i] + self.proc_rr[i]) as f64 * BINAURAL_SUM_GAIN;

                let peak = l.abs().max(r.abs());
                if peak > self.peak_env {
                    self.peak_env = peak; // instant attack
                } else {
                    self.peak_env *= self.limiter_release; // ~-12 dB/s release
                }

                let gain = if self.peak_env > 1.0 {
                    1.0 / self.peak_env
                } else {
                    1.0
                };
                self.out_buf[start + i * 2] = l * gain;
                self.out_buf[start + i * 2 + 1] = r * gain;
            }
            self.out_len += plen * 2;

            // Compact input
            self.in_left.copy_within(plen..self.in_len, 0);
            self.in_right.copy_within(plen..self.in_len, 0);
            self.in_len -= plen;
        }

        // Copy available output to data
        let avail = self.out_len.min(data.len());
        data[..avail].copy_from_slice(&self.out_buf[..avail]);

        // Zero-fill deficit (startup latency only — once the output backlog
        // covers a full callback this never runs again). Track the injected
        // amount so the player can delay its dry path to match.
        self.latency_samples += data.len() - avail;
        for s in &mut data[avail..] {
            *s = 0.0;
        }

        // Compact: shift unconsumed output to front
        let remaining = self.out_len - avail;
        if remaining > 0 {
            self.out_buf.copy_within(avail..self.out_len, 0);
        }
        self.out_len = remaining;
    }
}

static SOFA_44K: &[u8] = include_bytes!("../HRTF/D1_HRIR_SOFA/D1_44K_16bit_256tap_FIR_SOFA.sofa");
static SOFA_48K: &[u8] = include_bytes!("../HRTF/D1_HRIR_SOFA/D1_48K_24bit_256tap_FIR_SOFA.sofa");
static SOFA_96K: &[u8] = include_bytes!("../HRTF/D1_HRIR_SOFA/D1_96K_24bit_512tap_FIR_SOFA.sofa");

/// Load the two virtual-speaker HRIR filters, resampled to `sample_rate`.
///
/// Loads the SOFA dataset at its native rate (fast — no bulk resampling of the
/// entire measurement grid), extracts only the ±30° filters, then resamples
/// just those 4 short FIR vectors if the device rate differs.
fn load_filters(sample_rate: u32) -> Result<(Filter, Filter, usize), String> {
    let (data, native_rate): (&[u8], u32) = match sample_rate {
        ..=44100 => (SOFA_44K, 44100),
        44101..=72000 => (SOFA_48K, 48000),
        _ => (SOFA_96K, 96000),
    };

    let sofa = OpenOptions::new()
        .sample_rate(native_rate as f32)
        .open_data(data)
        .map_err(|e| format!("SOFA load: {e}"))?;

    let mut left_filter = Filter::new(sofa.filter_len());
    let mut right_filter = Filter::new(sofa.filter_len());
    sofa.filter(
        LEFT_SPEAKER.0,
        LEFT_SPEAKER.1,
        LEFT_SPEAKER.2,
        &mut left_filter,
    );
    sofa.filter(
        RIGHT_SPEAKER.0,
        RIGHT_SPEAKER.1,
        RIGHT_SPEAKER.2,
        &mut right_filter,
    );

    if sample_rate == native_rate {
        let len = sofa.filter_len();
        return Ok((left_filter, right_filter, len));
    }

    // Resample only the 4 extracted FIR tap vectors (not the whole dataset)
    left_filter.left = resample_fir(&left_filter.left, native_rate, sample_rate)?;
    left_filter.right = resample_fir(&left_filter.right, native_rate, sample_rate)?;
    right_filter.left = resample_fir(&right_filter.left, native_rate, sample_rate)?;
    right_filter.right = resample_fir(&right_filter.right, native_rate, sample_rate)?;

    let filter_len = left_filter.left.len();
    Ok((left_filter, right_filter, filter_len))
}

/// Resample a single FIR impulse response (f32 taps) with simple linear interpolation.
fn resample_fir(taps: &[f32], input_rate: u32, output_rate: u32) -> Result<Box<[f32]>, String> {
    if taps.is_empty() {
        return Ok(Vec::new().into_boxed_slice());
    }
    if input_rate == 0 || output_rate == 0 {
        return Err("FIR resample requires nonzero sample rates".into());
    }
    if input_rate == output_rate {
        return Ok(taps.to_vec().into_boxed_slice());
    }

    let output_len = crate::remaster::scaled_frame_count(taps.len(), input_rate, output_rate);
    let ratio = output_rate as f64 / input_rate as f64;
    let mut output = Vec::with_capacity(output_len);

    for frame in 0..output_len {
        let src = frame as f64 / ratio;
        let src_idx = src.floor() as usize;
        let next_idx = (src_idx + 1).min(taps.len() - 1);
        let frac = src - src_idx as f64;
        let a = taps[src_idx] as f64;
        let b = taps[next_idx] as f64;
        output.push((a * (1.0 - frac) + b * frac) as f32);
    }

    Ok(output.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::resample_fir;

    #[test]
    fn fir_resample_scales_length_and_stays_finite() {
        let taps = [0.0f32, 0.25, 1.0, 0.25, 0.0];
        let resampled = resample_fir(&taps, 44_100, 48_000).expect("FIR resample should work");

        assert_eq!(
            resampled.len(),
            crate::remaster::scaled_frame_count(taps.len(), 44_100, 48_000)
        );
        assert!(!resampled.is_empty());
        assert!(resampled.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn fir_resample_roundtrips_identity_rate() {
        let taps = [0.0f32, -0.5, 1.0, 0.25];
        let resampled = resample_fir(&taps, 48_000, 48_000).expect("identity FIR resample");

        assert_eq!(&*resampled, &taps);
    }
}
