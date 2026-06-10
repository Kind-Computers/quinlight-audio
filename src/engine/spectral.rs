// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(dead_code)]

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::simd;

fn deinterleave_channels(data: &[f64], channels: usize) -> Vec<Vec<f64>> {
    if channels == 2 {
        let (left, right) = simd::deinterleave_stereo_f64(data);
        return vec![left, right];
    }
    let frames = data.len() / channels;
    let mut separated = vec![Vec::with_capacity(frames); channels];
    for frame in data.chunks_exact(channels) {
        for (channel, &sample) in separated.iter_mut().zip(frame.iter()) {
            channel.push(sample);
        }
    }
    separated
}

/// Peak threshold below which a channel counts as digitally silent for
/// scoring purposes.
const SILENT_CHANNEL_PEAK: f64 = 1e-9;

/// True when both sides of a channel pair are digitally silent — a degenerate
/// but perfectly *matching* pair. Correlation is undefined there (zero
/// variance), and scoring it 0.0 would wrongly cap a stereo sample with one
/// silent channel at 0.5 forever (it could never pass the 0.9 gate no matter
/// how good the engines are). Such channels are neutral: skipped from the
/// average. One-sided silence is a genuine mismatch and still scores 0.0.
fn channel_pair_is_neutral(reference: &[f64], candidate: &[f64]) -> bool {
    let peak = |s: &[f64]| s.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    peak(reference) < SILENT_CHANNEL_PEAK && peak(candidate) < SILENT_CHANNEL_PEAK
}

/// Average per-channel scores, skipping neutral (both-silent) pairs. All
/// channels neutral means a silent sample reproduced exactly: a perfect match.
fn average_channel_scores(scores: &[Option<f64>]) -> f64 {
    let live: Vec<f64> = scores.iter().copied().flatten().collect();
    if live.is_empty() {
        return 1.0;
    }
    live.iter().sum::<f64>() / live.len() as f64
}

/// Compute the Pearson correlation of magnitude spectra between a reference
/// signal and a candidate, restricted to frequency bins below the original
/// sample's Nyquist frequency.
///
/// Both signals must be at 48kHz. For stereo, each channel is scored
/// independently against the matching reference channel and the final score is
/// the arithmetic mean of the per-channel correlations; channel pairs that are
/// silent on BOTH sides are neutral and excluded from the mean (see
/// [`channel_pair_is_neutral`]).
///
/// Returns a value in [-1.0, 1.0] where higher means the candidate better
/// preserves the original spectral content.
pub fn spectral_correlation(
    reference: &[f64],
    candidate: &[f64],
    channels: usize,
    original_rate: u32,
) -> f64 {
    if channels <= 1 {
        if channel_pair_is_neutral(reference, candidate) {
            return 1.0;
        }
        return spectral_correlation_channel(reference, candidate, original_rate);
    }

    let channel_scores: Vec<Option<f64>> = deinterleave_channels(reference, channels)
        .into_iter()
        .zip(deinterleave_channels(candidate, channels))
        .map(|(ref_ch, cand_ch)| {
            if channel_pair_is_neutral(&ref_ch, &cand_ch) {
                None
            } else {
                Some(spectral_correlation_channel(
                    &ref_ch,
                    &cand_ch,
                    original_rate,
                ))
            }
        })
        .collect();

    average_channel_scores(&channel_scores)
}

fn spectral_correlation_channel(reference: &[f64], candidate: &[f64], original_rate: u32) -> f64 {
    const OUTPUT_RATE: u32 = 48000;

    // Truncate to shorter length
    let len = reference.len().min(candidate.len());
    if len < 4 {
        return 0.0;
    }

    // Zero-pad to next power of 2
    let fft_len = len.next_power_of_two();

    let mut ref_buf: Vec<Complex<f64>> = reference[..len]
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut cand_buf: Vec<Complex<f64>> = candidate[..len]
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    // FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut ref_buf);
    fft.process(&mut cand_buf);

    // Determine how many bins to compare (up to original Nyquist)
    let nyquist_hz = original_rate as f64 / 2.0;
    let bin_hz = OUTPUT_RATE as f64 / fft_len as f64;
    let max_bin = ((nyquist_hz / bin_hz) as usize).min(fft_len / 2);
    if max_bin < 2 {
        return 0.0;
    }

    // Extract magnitude spectra
    let ref_mag: Vec<f64> = ref_buf[..max_bin].iter().map(|c| c.norm()).collect();
    let cand_mag: Vec<f64> = cand_buf[..max_bin].iter().map(|c| c.norm()).collect();

    // Pearson correlation
    simd::pearson_correlation(&ref_mag, &cand_mag)
}

#[cfg(test)]
/// Compute per-band energy ratios between two signals (mono, interleaved stereo
/// is deinterleaved externally).
///
/// Splits the spectrum into `num_bands` equal-width bands from 0 to
/// `max_freq_hz`.  Returns `Vec<(band_center_hz, energy_ratio_db)>` where
/// positive dB means the candidate has more energy in that band.
pub fn per_band_energy_ratio(
    reference: &[f64],
    candidate: &[f64],
    sample_rate: u32,
    max_freq_hz: f64,
    num_bands: usize,
) -> Vec<(f64, f64)> {
    let len = reference.len().min(candidate.len());
    if len < 4 || num_bands == 0 {
        return Vec::new();
    }

    let fft_len = len.next_power_of_two();

    let mut ref_buf: Vec<Complex<f64>> = reference[..len]
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut cand_buf: Vec<Complex<f64>> = candidate[..len]
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut ref_buf);
    fft.process(&mut cand_buf);

    let bin_hz = sample_rate as f64 / fft_len as f64;
    let band_width_hz = max_freq_hz / num_bands as f64;
    let max_bin = ((max_freq_hz / bin_hz) as usize).min(fft_len / 2);

    let mut results = Vec::with_capacity(num_bands);
    for band in 0..num_bands {
        let lo_hz = band as f64 * band_width_hz;
        let hi_hz = lo_hz + band_width_hz;
        let lo_bin = (lo_hz / bin_hz) as usize;
        let hi_bin = ((hi_hz / bin_hz) as usize).min(max_bin);

        if lo_bin >= hi_bin {
            results.push((lo_hz + band_width_hz / 2.0, 0.0));
            continue;
        }

        let ref_energy: f64 = ref_buf[lo_bin..hi_bin].iter().map(|c| c.norm_sqr()).sum();
        let cand_energy: f64 = cand_buf[lo_bin..hi_bin].iter().map(|c| c.norm_sqr()).sum();

        let ratio_db = if ref_energy > 1e-30 && cand_energy > 1e-30 {
            10.0 * (cand_energy / ref_energy).log10()
        } else {
            0.0
        };

        results.push((lo_hz + band_width_hz / 2.0, ratio_db));
    }

    results
}

const INTERSECTION_FFT_SIZE: usize = 2048;
const INTERSECTION_HOP: usize = 512; // 75% overlap
const INTERSECTION_SOFTMIN_EPS: f64 = 1e-12;
const STFT_REFERENCE_RATE_HZ: u32 = 48_000;

#[derive(Clone, Copy)]
struct StftConfig {
    fft_size: usize,
    hop: usize,
    sample_rate_hz: u32,
}

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (size - 1).max(1) as f64;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

fn magnitude_to_db(mag: f64) -> f64 {
    20.0 * mag.max(INTERSECTION_SOFTMIN_EPS).log10()
}

fn db_to_magnitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

fn window_magnitude_scale(window: &[f64]) -> f64 {
    1.0 / window.iter().sum::<f64>().max(INTERSECTION_SOFTMIN_EPS)
}

fn time_equivalent_stft_config(sample_rate_hz: u32) -> Option<StftConfig> {
    if sample_rate_hz < 2 {
        return None;
    }

    let scale = sample_rate_hz as f64 / STFT_REFERENCE_RATE_HZ as f64;
    let fft_size = (INTERSECTION_FFT_SIZE as f64 * scale).round() as usize;
    let hop = (INTERSECTION_HOP as f64 * scale).round() as usize;
    let fft_size = fft_size.max(16);
    let hop = hop.max(1).min(fft_size.saturating_sub(1).max(1));

    Some(StftConfig {
        fft_size,
        hop,
        sample_rate_hz,
    })
}

fn stft(signal: &[f64], fft_size: usize, hop: usize, window: &[f64]) -> Vec<Vec<Complex<f64>>> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut frames = Vec::new();
    let mut pos: usize = 0;
    while pos < signal.len() {
        let mut buf: Vec<Complex<f64>> = (0..fft_size)
            .map(|i| {
                let sample = if pos + i < signal.len() {
                    signal[pos + i]
                } else {
                    0.0
                };
                Complex::new(sample * window[i], 0.0)
            })
            .collect();
        fft.process(&mut buf);
        frames.push(buf);
        pos += hop;
    }
    frames
}

fn istft(
    frames: &[Vec<Complex<f64>>],
    fft_size: usize,
    hop: usize,
    window: &[f64],
    output_len: usize,
) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    let mut output = vec![0.0f64; output_len];
    let mut window_sum = vec![0.0f64; output_len];
    let scale = 1.0 / fft_size as f64;

    for (frame_idx, frame) in frames.iter().enumerate() {
        let pos = frame_idx * hop;
        let mut buf = frame.clone();
        ifft.process(&mut buf);
        for i in 0..fft_size {
            if pos + i < output_len {
                output[pos + i] += buf[i].re * scale * window[i];
                window_sum[pos + i] += window[i] * window[i];
            }
        }
    }

    for (sample, ws) in output.iter_mut().zip(window_sum.iter()) {
        if *ws > 1e-8 {
            *sample /= *ws;
        }
    }
    output
}

fn pad_for_stft(signal: &[f64], looped: bool, pad: usize) -> Vec<f64> {
    if signal.len() < 2 {
        return signal.to_vec();
    }
    let mut padded = Vec::with_capacity(signal.len() + 2 * pad);
    if looped {
        // Wrap circularly: prepend tail, append head — the STFT sees loop
        // continuity. Index arithmetic (rem_euclid) instead of slicing so
        // loops SHORTER than one pad/FFT frame — classic chip loops, the most
        // audibly-looped material there is — tile around as many times as
        // needed rather than silently falling back to reflection (which put a
        // mirror discontinuity exactly at the loop seam).
        let n = signal.len() as isize;
        for i in 0..pad {
            let idx = (i as isize - pad as isize).rem_euclid(n) as usize;
            padded.push(signal[idx]);
        }
        padded.extend_from_slice(signal);
        for i in 0..pad {
            padded.push(signal[i % signal.len()]);
        }
    } else {
        // Reflect at boundaries for non-looped or short signals
        for i in 0..pad {
            padded.push(signal[(pad - i).min(signal.len() - 1)]);
        }
        padded.extend_from_slice(signal);
        for i in 0..pad {
            let idx = signal.len().saturating_sub(2).saturating_sub(i);
            padded.push(signal[idx]);
        }
    }
    padded
}

fn absolute_bin_frequency_hz(bin: usize, fft_size: usize, sample_rate_hz: u32) -> f64 {
    if fft_size == 0 {
        return 0.0;
    }
    let mirrored_bin = bin.min(fft_size.saturating_sub(bin));
    mirrored_bin as f64 * sample_rate_hz as f64 / fft_size as f64
}

/// Per-STFT-frame energy summed over bins whose center frequency exceeds
/// `hf_cutoff_hz`. One value per STFT frame; length depends on `data.len()`
/// and the STFT config for `sample_rate_hz`. Used by the loop quality gate
/// to localize AI-added HF content vs. its sample-wide average.
pub(crate) fn hf_energy_envelope(data: &[f64], sample_rate_hz: u32, hf_cutoff_hz: f64) -> Vec<f64> {
    if data.len() < 4 || !hf_cutoff_hz.is_finite() || hf_cutoff_hz < 0.0 {
        return Vec::new();
    }
    let Some(cfg) = time_equivalent_stft_config(sample_rate_hz) else {
        return Vec::new();
    };
    let window = hann_window(cfg.fft_size);
    let frames = stft(data, cfg.fft_size, cfg.hop, &window);
    if frames.is_empty() {
        return Vec::new();
    }
    let nyquist_bin = cfg.fft_size / 2;
    let mut envelope = Vec::with_capacity(frames.len());
    for frame in &frames {
        let mut hf_energy = 0.0f64;
        for (k, bin) in frame.iter().enumerate().take(nyquist_bin + 1) {
            let freq = absolute_bin_frequency_hz(k, cfg.fft_size, sample_rate_hz);
            if freq > hf_cutoff_hz {
                let mag = bin.norm();
                hf_energy += mag * mag;
            }
        }
        envelope.push(hf_energy);
    }
    envelope
}

/// Maps a sample-domain index to an STFT frame index under the config used
/// by `hf_energy_envelope` at the same rate. Returns 0 when the hop is
/// degenerate.
pub(crate) fn stft_frame_for_sample(sample_idx: usize, sample_rate_hz: u32) -> usize {
    match time_equivalent_stft_config(sample_rate_hz) {
        Some(cfg) if cfg.hop > 0 => sample_idx / cfg.hop,
        _ => 0,
    }
}

// Returns (w_source, w_candidate) so `w_source + w_candidate == 1` across the
// band. Linear weights preserve magnitude when source and candidate carry the
// same content in-phase (common in sinc-resample fallbacks), while still
// giving 100% source at DC and 100% candidate at the source Nyquist. Equal-
// power (cos/sin) was considered but would boost in-phase correlated inputs
// by √2 at midband, showing up as extra HF energy in sinc-fallback paths.
fn source_blend_weights(freq_hz: f64, source_nyquist_hz: f64) -> (f64, f64) {
    if !freq_hz.is_finite() || !source_nyquist_hz.is_finite() || source_nyquist_hz <= 0.0 {
        return (1.0, 0.0);
    }
    let t = (freq_hz.max(0.0) / source_nyquist_hz).clamp(0.0, 1.0);
    (1.0 - t, t)
}

fn frame_center_seconds(frame_index: usize, cfg: StftConfig) -> f64 {
    ((frame_index * cfg.hop) as f64 + cfg.fft_size as f64 * 0.5) / cfg.sample_rate_hz as f64
}

fn source_frame_position_for_candidate(
    candidate_frame_index: usize,
    candidate_cfg: StftConfig,
    source_cfg: StftConfig,
) -> f64 {
    let candidate_center = frame_center_seconds(candidate_frame_index, candidate_cfg);
    ((candidate_center * source_cfg.sample_rate_hz as f64) - source_cfg.fft_size as f64 * 0.5)
        / source_cfg.hop as f64
}

fn interpolate_positive_spectrum_magnitude(
    magnitudes: &[f64],
    freq_hz: f64,
    cfg: StftConfig,
) -> f64 {
    if magnitudes.is_empty() || cfg.sample_rate_hz < 2 {
        return 0.0;
    }
    let max_bin = magnitudes.len().saturating_sub(1);
    let bin_pos = (freq_hz.max(0.0) * cfg.fft_size as f64 / cfg.sample_rate_hz as f64)
        .clamp(0.0, max_bin as f64);
    let lo = bin_pos.floor() as usize;
    let hi = (lo + 1).min(max_bin);
    let mix = bin_pos - lo as f64;
    magnitudes[lo] * (1.0 - mix) + magnitudes[hi] * mix
}

fn interpolate_source_magnitude(
    source_frames: &[Vec<f64>],
    frame_pos: f64,
    freq_hz: f64,
    source_cfg: StftConfig,
) -> f64 {
    if source_frames.is_empty() {
        return 0.0;
    }
    let max_frame = source_frames.len().saturating_sub(1);
    let clamped = frame_pos.clamp(0.0, max_frame as f64);
    let lo = clamped.floor() as usize;
    let hi = (lo + 1).min(max_frame);
    let mix = clamped - lo as f64;
    let lo_mag = interpolate_positive_spectrum_magnitude(&source_frames[lo], freq_hz, source_cfg);
    let hi_mag = interpolate_positive_spectrum_magnitude(&source_frames[hi], freq_hz, source_cfg);
    lo_mag * (1.0 - mix) + hi_mag * mix
}

pub(crate) fn spectral_correlation_across_rates(
    reference: &[f64],
    reference_rate_hz: u32,
    candidate: &[f64],
    candidate_rate_hz: u32,
    channels: usize,
    looped: bool,
) -> f64 {
    if channels <= 1 {
        if channel_pair_is_neutral(reference, candidate) {
            return 1.0;
        }
        return spectral_correlation_across_rates_channel(
            reference,
            reference_rate_hz,
            candidate,
            candidate_rate_hz,
            looped,
        );
    }

    let channel_scores: Vec<Option<f64>> = deinterleave_channels(reference, channels)
        .into_iter()
        .zip(deinterleave_channels(candidate, channels))
        .map(|(ref_ch, cand_ch)| {
            if channel_pair_is_neutral(&ref_ch, &cand_ch) {
                None
            } else {
                Some(spectral_correlation_across_rates_channel(
                    &ref_ch,
                    reference_rate_hz,
                    &cand_ch,
                    candidate_rate_hz,
                    looped,
                ))
            }
        })
        .collect();

    average_channel_scores(&channel_scores)
}

fn spectral_correlation_across_rates_channel(
    reference: &[f64],
    reference_rate_hz: u32,
    candidate: &[f64],
    candidate_rate_hz: u32,
    looped: bool,
) -> f64 {
    if reference.len() < 4 || candidate.len() < 4 || reference_rate_hz < 2 || candidate_rate_hz < 2
    {
        return 0.0;
    }

    let Some(reference_cfg) = time_equivalent_stft_config(reference_rate_hz) else {
        return 0.0;
    };
    let Some(candidate_cfg) = time_equivalent_stft_config(candidate_rate_hz) else {
        return 0.0;
    };

    let reference_padded = pad_for_stft(reference, looped, reference_cfg.fft_size);
    let candidate_padded = pad_for_stft(candidate, looped, candidate_cfg.fft_size);
    let reference_window = hann_window(reference_cfg.fft_size);
    let candidate_window = hann_window(candidate_cfg.fft_size);
    let reference_magnitude_scale = window_magnitude_scale(&reference_window);
    let candidate_magnitude_scale = window_magnitude_scale(&candidate_window);

    let reference_frames = stft(
        &reference_padded,
        reference_cfg.fft_size,
        reference_cfg.hop,
        &reference_window,
    );
    if reference_frames.is_empty() {
        return 0.0;
    }

    let candidate_frames = stft(
        &candidate_padded,
        candidate_cfg.fft_size,
        candidate_cfg.hop,
        &candidate_window,
    );
    if candidate_frames.is_empty() {
        return 0.0;
    }

    let reference_magnitudes: Vec<Vec<f64>> = reference_frames
        .iter()
        .map(|frame| {
            frame[..=reference_cfg.fft_size / 2]
                .iter()
                .map(|bin| bin.norm() * reference_magnitude_scale)
                .collect()
        })
        .collect();

    let max_freq_hz = reference_rate_hz.min(candidate_rate_hz) as f64 * 0.5;
    let mut reference_bins = Vec::new();
    let mut candidate_bins = Vec::new();

    for (frame_index, frame) in candidate_frames.iter().enumerate() {
        let reference_frame_pos =
            source_frame_position_for_candidate(frame_index, candidate_cfg, reference_cfg);
        for (bin_index, bin) in frame[..=candidate_cfg.fft_size / 2].iter().enumerate() {
            let freq_hz =
                absolute_bin_frequency_hz(bin_index, candidate_cfg.fft_size, candidate_rate_hz);
            if freq_hz > max_freq_hz {
                break;
            }

            let candidate_mag = bin.norm() * candidate_magnitude_scale;
            let reference_mag = interpolate_source_magnitude(
                &reference_magnitudes,
                reference_frame_pos,
                freq_hz,
                reference_cfg,
            );
            if candidate_mag <= INTERSECTION_SOFTMIN_EPS
                && reference_mag <= INTERSECTION_SOFTMIN_EPS
            {
                continue;
            }
            reference_bins.push(reference_mag);
            candidate_bins.push(candidate_mag);
        }
    }

    if reference_bins.len() < 4 {
        return 0.0;
    }

    simd::pearson_correlation(&reference_bins, &candidate_bins)
}

// Phase-aware spectral blend between the upscaled candidate and the original
// source, both expressed at the same sample rate so their STFTs share an
// identical bin and frame structure. Below the source's original Nyquist the
// candidate bin is mixed toward the source bin via `rotor::polar_lerp` —
// arithmetic-mean magnitude (preserves the crossfade's energy-conservation
// expectation: pure source at f=0, pure candidate at f=source_nyquist) with
// shortest-arc SLERP on the phase axis. Above the source Nyquist the source
// has no usable content, so the candidate passes through unchanged. At DC
// the blend reads fully from the source — phase is anchored to the original
// where loop-seam visibility is dominated by slow bass — fading linearly
// toward 100% candidate at the source Nyquist. The caller resamples
// `source_at_output_rate` to the candidate's sample rate so bin phases line
// up; `source_original_rate_hz` is only used to locate the Nyquist crossover.
//
// Why SLERP on the phase axis (vs Cartesian complex lerp): at bins where
// source and candidate disagree on phase, the chord through ℂ between two
// phasors of comparable magnitude passes closer to the origin than either
// endpoint. That hidden per-bin attenuation depends on per-bin phase
// agreement, which varies discontinuously across frequency — so the
// inverse STFT renders it as pre-echo and transient smearing in the time
// domain. SLERP on `S¹` keeps phase on the geodesic so a phase
// disagreement at a bin doesn't silently shrink that bin's energy.
pub(crate) fn apply_source_frequency_blend(
    candidate: &[f64],
    source_at_output_rate: &[f64],
    source_original_rate_hz: u32,
    output_rate_hz: u32,
    looped: bool,
) -> Vec<f64> {
    if candidate.len() < 4
        || source_at_output_rate.len() < 4
        || source_original_rate_hz < 2
        || output_rate_hz < 2
    {
        return candidate.to_vec();
    }

    let Some(cfg) = time_equivalent_stft_config(output_rate_hz) else {
        return candidate.to_vec();
    };

    let candidate_padded = pad_for_stft(candidate, looped, cfg.fft_size);
    let source_padded = pad_for_stft(source_at_output_rate, looped, cfg.fft_size);
    let window = hann_window(cfg.fft_size);

    let mut candidate_frames = stft(&candidate_padded, cfg.fft_size, cfg.hop, &window);
    if candidate_frames.is_empty() {
        return candidate.to_vec();
    }
    let source_frames = stft(&source_padded, cfg.fft_size, cfg.hop, &window);
    if source_frames.is_empty() {
        return candidate.to_vec();
    }

    let source_nyquist_hz = (source_original_rate_hz as f64 * 0.5).min(output_rate_hz as f64 * 0.5);
    let nyquist_bin = cfg.fft_size / 2;

    for (frame_index, frame) in candidate_frames.iter_mut().enumerate() {
        // Clamp so a candidate longer than the source keeps reading the last
        // source frame rather than reading off the end.
        let src_frame_idx = frame_index.min(source_frames.len().saturating_sub(1));
        let src_frame = &source_frames[src_frame_idx];
        for k in 0..=nyquist_bin {
            let freq_hz = absolute_bin_frequency_hz(k, cfg.fft_size, output_rate_hz);
            if freq_hz >= source_nyquist_hz {
                // Source carries no content above its Nyquist; leave the
                // candidate bin untouched and let the conjugate half stay
                // in sync via the real-input FFT.
                continue;
            }

            let (w_src, w_cand) = source_blend_weights(freq_hz, source_nyquist_hz);
            // (w_src, w_cand) sums to 1 by construction, so w_cand is the
            // crossfade parameter t (t=0 → pure source, t=1 → pure
            // candidate). polar_lerp gives arithmetic-mean magnitude
            // (audio crossfade convention) with SLERP phase.
            let _ = w_src;
            let new_bin = crate::engine::rotor::polar_lerp(src_frame[k], frame[k], w_cand);
            frame[k] = new_bin;

            if k > 0 && k < nyquist_bin {
                frame[cfg.fft_size - k] = new_bin.conj();
            }
        }
    }

    let full = istft(
        &candidate_frames,
        cfg.fft_size,
        cfg.hop,
        &window,
        candidate_padded.len(),
    );
    let start = cfg.fft_size.min(full.len());
    let end = (cfg.fft_size + candidate.len()).min(full.len());
    full[start..end].to_vec()
}

/// Spectral intersection of multiple engine outputs.
///
/// Per-bin Karcher mean on the rotor manifold (ℝ⁺ × S¹):
/// - **Magnitude**: geometric mean of engine magnitudes — Karcher mean on
///   (ℝ⁺, ·). Smooth in every input; biased toward smaller magnitudes
///   (preserves softmin's "trust the quietest engine" intent without the
///   per-bin discrete-winner ringing of patched-together spectra).
/// - **Phase**: circular mean of engine phases — Karcher mean on S¹.
/// - **Agreement scaling**: the resultant length of the phase rotor sum
///   (0–1) attenuates the magnitude proportionally, so bins where engines
///   disagree on direction (the typical hallucination fingerprint) get
///   suppressed.
///
/// When `looped` is true, the signal is wrapped at boundaries so the STFT sees
/// loop continuity, eliminating edge artifacts at the loop point.
///
/// Operates on single-channel (mono) signals. For multi-channel audio, call
/// once per channel.
pub fn spectral_intersection(engine_signals: &[&[f64]], looped: bool) -> Vec<f64> {
    if engine_signals.is_empty() {
        return Vec::new();
    }
    if engine_signals.len() == 1 {
        return engine_signals[0].to_vec();
    }

    let output_len = engine_signals.iter().map(|s| s.len()).min().unwrap_or(0);
    if output_len < 4 {
        // Too short for any spectral analysis: pass the first engine through
        // rather than destructively forcing silence.
        return engine_signals[0][..output_len].to_vec();
    }

    let pad = INTERSECTION_FFT_SIZE;
    let padded_signals: Vec<Vec<f64>> = engine_signals
        .iter()
        .map(|sig| pad_for_stft(&sig[..output_len], looped, INTERSECTION_FFT_SIZE))
        .collect();
    let padded_len = padded_signals[0].len();

    let window = hann_window(INTERSECTION_FFT_SIZE);
    let all_stfts: Vec<Vec<Vec<Complex<f64>>>> = padded_signals
        .iter()
        .map(|sig| stft(sig, INTERSECTION_FFT_SIZE, INTERSECTION_HOP, &window))
        .collect();

    let num_frames = all_stfts.iter().map(|s| s.len()).min().unwrap_or(0);
    let n_engines = all_stfts.len() as f64;

    let mut consensus_frames: Vec<Vec<Complex<f64>>> = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let num_bins = all_stfts[0][frame_idx].len();
        let mut out_frame = Vec::with_capacity(num_bins);
        let mut engine_magnitudes = Vec::with_capacity(all_stfts.len());
        for bin in 0..num_bins {
            engine_magnitudes.clear();
            engine_magnitudes.extend(all_stfts.iter().map(|engine| engine[frame_idx][bin].norm()));
            // Karcher mean on (ℝ⁺, ·). Smooth in every input; biased toward
            // smaller magnitudes (the rotor-correct successor to softmin
            // without the per-bin discrete-winner ringing).
            let mut consensus_mag =
                crate::engine::rotor::geometric_mean_magnitude(&engine_magnitudes);

            let (sum_cos, sum_sin) = all_stfts.iter().fold((0.0, 0.0), |(c, s), engine| {
                let phase = engine[frame_idx][bin].arg();
                (c + phase.cos(), s + phase.sin())
            });
            let mean_cos = sum_cos / n_engines;
            let mean_sin = sum_sin / n_engines;
            let agreement = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
            consensus_mag *= agreement;
            let phase = mean_sin.atan2(mean_cos);

            out_frame.push(Complex::from_polar(consensus_mag, phase));
        }
        consensus_frames.push(out_frame);
    }

    let full = istft(
        &consensus_frames,
        INTERSECTION_FFT_SIZE,
        INTERSECTION_HOP,
        &window,
        padded_len,
    );

    // Trim padding to recover original-length signal
    let start = pad.min(full.len());
    let end = (pad + output_len).min(full.len());
    full[start..end].to_vec()
}

/// Cross-correlation of `template` against `signal` via FFT, scaled by the
/// template's energy so that a perfect in-phase match at unit signal gain
/// yields `1.0`.
///
/// Returns one value per lag, length = `fft_len` (the next power of two ≥
/// `template.len() + signal.len() - 1`). The peak value indicates the lag
/// at which `template` best matches `signal`.
///
/// The output is the raw linear cross-correlation divided by the
/// template's squared L2 norm (Σ template[i]²). When `signal = g *
/// template` at the matching lag, the peak is `g` — so the value
/// reflects both match quality AND the local signal gain. Callers that
/// need a truly gain-invariant match score must additionally divide by
/// the square root of the local signal energy over the template window
/// (proper normalized cross-correlation). See `select_best_body_copy_in_block`
/// in `remaster.rs` for the canonical NCC-correction pattern.
pub(crate) fn fft_cross_correlation(template: &[f64], signal: &[f64]) -> Vec<f64> {
    if template.is_empty() || signal.is_empty() {
        return Vec::new();
    }

    let fft_len = (template.len() + signal.len() - 1).next_power_of_two();

    let mut tmpl_buf: Vec<Complex<f64>> = template
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut sig_buf: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    fft_fwd.process(&mut tmpl_buf);
    fft_fwd.process(&mut sig_buf);

    let mut product: Vec<Complex<f64>> = sig_buf
        .iter()
        .zip(tmpl_buf.iter())
        .map(|(s, t)| s * t.conj())
        .collect();

    let fft_inv = planner.plan_fft_inverse(fft_len);
    fft_inv.process(&mut product);

    // Exact match cross-correlation peak = Σ template[i]² (the template's
    // squared L2 norm). Dividing by that yields peak = 1.0 at perfect
    // match, independent of amplitude. The 1/fft_len factor undoes
    // rustfft's IFFT scaling convention (it doesn't normalize).
    let tmpl_energy_sq = template.iter().map(|x| x * x).sum::<f64>().max(1e-30);
    let scale = 1.0 / (fft_len as f64 * tmpl_energy_sq);

    product.iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn rms(signal: &[f64]) -> f64 {
        (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt()
    }

    /// Regression: a stereo sample with one digitally-silent channel used to
    /// score 0.0 on that channel even when the candidate matched exactly,
    /// capping the mean at 0.5 — below the 0.9 gate forever. Both-silent
    /// channel pairs are neutral (skipped); one-sided silence still fails.
    #[test]
    fn silent_channel_pair_is_neutral_not_zero() {
        let frames = 4096;
        let left = tone(48_000, frames, 1000.0, 0.5, 0.0);
        // Interleave: left = tone, right = silence, candidate identical.
        let mut stereo = Vec::with_capacity(frames * 2);
        for &l in &left {
            stereo.push(l);
            stereo.push(0.0);
        }
        let score = spectral_correlation(&stereo, &stereo, 2, 22_050);
        assert!(
            score > 0.95,
            "identical stereo with a silent channel must score ~1.0, got {score}"
        );

        let score_xrate =
            spectral_correlation_across_rates(&stereo, 22_050, &stereo, 22_050, 2, false);
        assert!(
            score_xrate > 0.95,
            "cross-rate path must also treat both-silent channels as neutral, got {score_xrate}"
        );

        // One-sided silence is a genuine mismatch: candidate puts noise on
        // the silent channel.
        let mut noisy = stereo.clone();
        for frame in 0..frames {
            noisy[frame * 2 + 1] = if frame % 2 == 0 { 0.1 } else { -0.1 };
        }
        let score_mismatch = spectral_correlation(&stereo, &noisy, 2, 22_050);
        assert!(
            score_mismatch < 0.9,
            "noise on a silent reference channel must drag the score down, got {score_mismatch}"
        );

        // Fully silent sample reproduced exactly: perfect match by definition.
        let silence = vec![0.0f64; frames * 2];
        assert_eq!(spectral_correlation(&silence, &silence, 2, 22_050), 1.0);
    }

    /// Regression: looped signals shorter than one pad length used to fall
    /// back to reflect padding — a mirror discontinuity exactly at the loop
    /// seam, defeating the `looped` flag for classic short chip loops.
    #[test]
    fn pad_for_stft_wraps_circularly_for_loops_shorter_than_pad() {
        let signal: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let pad = 32usize;
        let padded = pad_for_stft(&signal, true, pad);
        assert_eq!(padded.len(), signal.len() + 2 * pad);
        // Every padded sample must equal the circular extension of the loop.
        for (k, &x) in padded.iter().enumerate() {
            let idx = (k as isize - pad as isize).rem_euclid(signal.len() as isize) as usize;
            assert_eq!(
                x, signal[idx],
                "padded[{k}] must wrap circularly, got {x} expected {}",
                signal[idx]
            );
        }
    }

    /// The long-loop wrap behavior is unchanged by the short-loop fix.
    #[test]
    fn pad_for_stft_long_loop_matches_slice_wrap() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let pad = 16usize;
        let padded = pad_for_stft(&signal, true, pad);
        let mut expected = Vec::new();
        expected.extend_from_slice(&signal[signal.len() - pad..]);
        expected.extend_from_slice(&signal);
        expected.extend_from_slice(&signal[..pad]);
        assert_eq!(padded, expected);
    }

    fn tone(
        sample_rate_hz: u32,
        frames: usize,
        freq_hz: f64,
        amplitude: f64,
        phase: f64,
    ) -> Vec<f64> {
        (0..frames)
            .map(|i| {
                let t = i as f64 / sample_rate_hz as f64;
                amplitude * (2.0 * std::f64::consts::PI * freq_hz * t + phase).sin()
            })
            .collect()
    }

    fn mix(signals: &[Vec<f64>]) -> Vec<f64> {
        let len = signals.iter().map(|signal| signal.len()).min().unwrap_or(0);
        let mut mixed = vec![0.0; len];
        for signal in signals {
            for (out, &sample) in mixed.iter_mut().zip(signal.iter()) {
                *out += sample;
            }
        }
        mixed
    }

    fn component_amplitude(signal: &[f64], freq_hz: f64, sample_rate_hz: u32) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        let mut sin_dot = 0.0;
        let mut cos_dot = 0.0;
        for (idx, &sample) in signal.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * freq_hz * idx as f64 / sample_rate_hz as f64;
            sin_dot += sample * phase.sin();
            cos_dot += sample * phase.cos();
        }
        2.0 * (sin_dot * sin_dot + cos_dot * cos_dot).sqrt() / signal.len() as f64
    }

    #[test]
    fn identical_signals_have_correlation_one() {
        let signal: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = spectral_correlation(&signal, &signal, 1, 8000);
        assert!(r > 0.999, "Expected ~1.0, got {r}");
    }

    #[test]
    fn uncorrelated_signals_have_low_correlation() {
        let signal_a: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
        let signal_b: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.73).cos()).collect();
        let r = spectral_correlation(&signal_a, &signal_b, 1, 8000);
        assert!(r < 0.5, "Expected low correlation, got {r}");
    }

    #[test]
    fn identical_stereo_channels_score_near_one() {
        let mono: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).sin()).collect();
        let stereo: Vec<f64> = mono.iter().flat_map(|&s| [s, s].into_iter()).collect();
        let r = spectral_correlation(&stereo, &stereo, 2, 8000);
        assert!(r > 0.999, "Expected ~1.0 for identical stereo, got {r}");
    }

    #[test]
    fn stereo_scores_are_averaged_per_channel() {
        let left: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
        let right_reference: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.23).cos()).collect();
        let right_candidate: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.77).sin()).collect();

        let stereo_reference: Vec<f64> = left
            .iter()
            .zip(right_reference.iter())
            .flat_map(|(&l, &r)| [l, r])
            .collect();
        let stereo_candidate: Vec<f64> = left
            .iter()
            .zip(right_candidate.iter())
            .flat_map(|(&l, &r)| [l, r])
            .collect();

        let stereo_score = spectral_correlation(&stereo_reference, &stereo_candidate, 2, 8000);
        let left_score = spectral_correlation(&left, &left, 1, 8000);
        let right_score = spectral_correlation(&right_reference, &right_candidate, 1, 8000);
        let expected = (left_score + right_score) * 0.5;

        assert!(
            (stereo_score - expected).abs() < 1e-6,
            "Expected stereo score {stereo_score} to equal mean per-channel score {expected}",
        );
    }

    #[test]
    fn cross_rate_correlation_prefers_faithful_8khz_candidate_over_in_band_artifact() {
        let source_rate = 8_000;
        let candidate_rate = 48_000;
        let source_frames = 4096;
        let candidate_frames =
            (source_frames as f64 * candidate_rate as f64 / source_rate as f64).round() as usize;
        let source = mix(&[
            tone(source_rate, source_frames, 440.0, 1.0, 0.0),
            tone(source_rate, source_frames, 1_700.0, 0.35, 0.3),
        ]);
        let faithful = mix(&[
            tone(candidate_rate, candidate_frames, 440.0, 1.0, 0.0),
            tone(candidate_rate, candidate_frames, 1_700.0, 0.35, 0.3),
        ]);
        let distorted = mix(&[
            faithful.clone(),
            tone(candidate_rate, candidate_frames, 2_500.0, 0.60, 0.0),
        ]);

        let faithful_score = spectral_correlation_across_rates(
            &source,
            source_rate,
            &faithful,
            candidate_rate,
            1,
            false,
        );
        let distorted_score = spectral_correlation_across_rates(
            &source,
            source_rate,
            &distorted,
            candidate_rate,
            1,
            false,
        );

        assert!(
            faithful_score > 0.85,
            "Faithful cross-rate candidate should score highly, got {faithful_score:.4}"
        );
        assert!(
            faithful_score > distorted_score + 0.10,
            "In-band artifact should reduce cross-rate score \
             (faithful={faithful_score:.4}, distorted={distorted_score:.4})"
        );
    }

    #[test]
    fn cross_rate_correlation_prefers_faithful_22khz_candidate_over_mismatched_spectrum() {
        let source_rate = 22_050;
        let candidate_rate = 48_000;
        let source_frames = 4096;
        let candidate_frames =
            (source_frames as f64 * candidate_rate as f64 / source_rate as f64).round() as usize;
        let source = mix(&[
            tone(source_rate, source_frames, 330.0, 0.8, 0.0),
            tone(source_rate, source_frames, 6_200.0, 0.4, 0.25),
        ]);
        let faithful = mix(&[
            tone(candidate_rate, candidate_frames, 330.0, 0.8, 0.0),
            tone(candidate_rate, candidate_frames, 6_200.0, 0.4, 0.25),
        ]);
        let mismatched = mix(&[
            tone(candidate_rate, candidate_frames, 330.0, 0.8, 0.0),
            tone(candidate_rate, candidate_frames, 5_100.0, 0.4, 0.25),
        ]);

        let faithful_score = spectral_correlation_across_rates(
            &source,
            source_rate,
            &faithful,
            candidate_rate,
            1,
            false,
        );
        let mismatched_score = spectral_correlation_across_rates(
            &source,
            source_rate,
            &mismatched,
            candidate_rate,
            1,
            false,
        );

        assert!(
            faithful_score > 0.80,
            "Faithful 22.05 kHz cross-rate candidate should score highly, got {faithful_score:.4}"
        );
        assert!(
            faithful_score > mismatched_score + 0.15,
            "Mismatched upper-band content should score worse \
             (faithful={faithful_score:.4}, mismatched={mismatched_score:.4})"
        );
    }

    #[test]
    fn stft_istft_roundtrip() {
        let n = 32768;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let window = hann_window(INTERSECTION_FFT_SIZE);
        let frames = stft(&signal, INTERSECTION_FFT_SIZE, INTERSECTION_HOP, &window);
        let reconstructed = istft(
            &frames,
            INTERSECTION_FFT_SIZE,
            INTERSECTION_HOP,
            &window,
            signal.len(),
        );
        // Check interior only — raw STFT edges lack full overlap coverage
        let margin = INTERSECTION_FFT_SIZE;
        let max_err: f64 = signal[margin..n - margin]
            .iter()
            .zip(reconstructed[margin..n - margin].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-4, "STFT roundtrip error too large: {max_err}");
    }

    #[test]
    fn padded_roundtrip_covers_full_signal() {
        let n = 4096;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        // spectral_intersection with identical inputs is effectively a padded roundtrip
        let result = spectral_intersection(&[&signal, &signal], false);
        assert_eq!(result.len(), n);
        let max_err: f64 = signal
            .iter()
            .zip(result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 0.05,
            "Padded roundtrip should reconstruct full signal, max_err={max_err}"
        );
    }

    #[test]
    fn intersection_looped_preserves_loop_continuity() {
        let n = 4096;
        let sr = 48000.0;
        let freq = 440.0;
        // Create a signal that's continuous at the loop point
        // (integer number of cycles so start and end match)
        let cycles = (freq * n as f64 / sr).round();
        let actual_freq = cycles * sr / n as f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * actual_freq * i as f64 / sr).sin())
            .collect();
        // Add slight per-engine variation
        let engine_a: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.02 * (i as f64 * 3.7).sin())
            .collect();
        let engine_b: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.02 * (i as f64 * 5.1).cos())
            .collect();
        let result = spectral_intersection(&[&engine_a, &engine_b], true);
        // Check loop continuity: last sample should be close to first sample
        let gap = (result[result.len() - 1] - result[0]).abs();
        let peak = result.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let relative_gap = if peak > 0.0 { gap / peak } else { 0.0 };
        assert!(
            relative_gap < 0.1,
            "Loop point should be continuous, relative gap={relative_gap} (gap={gap}, peak={peak})"
        );
    }

    #[test]
    fn intersection_preserves_shared_frequency() {
        // Both engines produce the same 440 Hz sine
        let n = 8192;
        let freq = 440.0;
        let sr = 48000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let result = spectral_intersection(&[&signal, &signal], false);
        let rms_in: f64 = (signal.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        let rms_out: f64 = (result.iter().map(|x| x * x).sum::<f64>() / result.len() as f64).sqrt();
        let ratio = rms_out / rms_in;
        assert!(
            ratio > 0.8,
            "Shared frequency should be largely preserved, got ratio {ratio}"
        );
    }

    #[test]
    fn intersection_geomean_lifts_conservative_engine_without_matching_strongest() {
        let n = 8192;
        let sr = 48000.0;
        let freq = 440.0;
        let strong: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect();
        let weak: Vec<f64> = strong.iter().map(|&s| s * 0.3).collect();

        let result = spectral_intersection(&[&strong, &weak], false);
        let rms_strong = rms(&strong);
        let rms_weak = rms(&weak);
        let rms_result = rms(&result);

        assert!(
            rms_result > rms_weak * 1.2,
            "Soft-min should lift suppressed shared energy above hard-min behavior: \
             weak_rms={rms_weak:.4}, result_rms={rms_result:.4}"
        );
        assert!(
            rms_result < rms_strong * 0.95,
            "Soft-min should stay conservative and below the strongest engine: \
             strong_rms={rms_strong:.4}, result_rms={rms_result:.4}"
        );
    }

    #[test]
    fn intersection_rejects_unshared_frequency() {
        let n = 8192;
        let sr = 48000.0;
        let shared_freq = 440.0;
        let hallucinated_freq = 3000.0;
        // Engine A: shared + hallucinated
        let engine_a: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * shared_freq * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * hallucinated_freq * t).sin()
            })
            .collect();
        // Engine B: shared only
        let engine_b: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * shared_freq * t).sin()
            })
            .collect();
        let result = spectral_intersection(&[&engine_a, &engine_b], false);
        // Measure energy of the hallucinated frequency in the result via correlation
        let hallucinated_ref: Vec<f64> = (0..result.len())
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * hallucinated_freq * t).sin()
            })
            .collect();
        let dot: f64 = result
            .iter()
            .zip(hallucinated_ref.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_h: f64 = hallucinated_ref.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_r: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        let correlation = if norm_h > 0.0 && norm_r > 0.0 {
            dot / (norm_h * norm_r)
        } else {
            0.0
        };
        assert!(
            correlation.abs() < 0.15,
            "Hallucinated frequency should be rejected, got correlation {correlation}"
        );
    }

    #[test]
    fn intersection_geomean_rejects_hallucinated_frequency_better_than_mean_blend() {
        let n = 8192;
        let sr = 48000.0;
        let shared_freq = 440.0;
        let hallucinated_freq = 3000.0;
        let engine_a: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * shared_freq * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * hallucinated_freq * t).sin()
            })
            .collect();
        let engine_b: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * shared_freq * t).sin()
            })
            .collect();

        let result = spectral_intersection(&[&engine_a, &engine_b], false);
        let mean_blend: Vec<f64> = engine_a
            .iter()
            .zip(engine_b.iter())
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let hallucinated_ref: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * hallucinated_freq * t).sin()
            })
            .collect();

        let hallucination_correlation = |signal: &[f64]| -> f64 {
            let dot: f64 = signal
                .iter()
                .zip(hallucinated_ref.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_h: f64 = hallucinated_ref.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_s: f64 = signal.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_h > 0.0 && norm_s > 0.0 {
                dot / (norm_h * norm_s)
            } else {
                0.0
            }
        };

        let result_corr = hallucination_correlation(&result).abs();
        let mean_corr = hallucination_correlation(&mean_blend).abs();

        assert!(
            result_corr < 0.15,
            "Soft-min should still suppress hallucinated energy, got correlation {result_corr}"
        );
        assert!(
            result_corr < mean_corr * 0.5,
            "Soft-min should reject hallucinated-only content more strongly than a mean blend: \
             result_corr={result_corr:.4}, mean_corr={mean_corr:.4}"
        );
    }

    #[test]
    fn intersection_three_engines() {
        let n = 8192;
        let sr = 48000.0;
        let shared_freq = 1000.0;
        let signals: Vec<Vec<f64>> = (0..3)
            .map(|engine_idx| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        let shared = (2.0 * std::f64::consts::PI * shared_freq * t).sin();
                        // Each engine adds a unique hallucinated frequency
                        let hallucinated_freq = 2000.0 + engine_idx as f64 * 1500.0;
                        shared + 0.3 * (2.0 * std::f64::consts::PI * hallucinated_freq * t).sin()
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = signals.iter().map(|s| s.as_slice()).collect();
        let result = spectral_intersection(&refs, false);
        let rms: f64 = (result.iter().map(|x| x * x).sum::<f64>() / result.len() as f64).sqrt();
        assert!(
            rms > 0.1,
            "Result should have substantial energy from shared frequency"
        );
    }

    #[test]
    fn circular_mean_sets_consensus_phase_below_original_nyquist() {
        let n = 8192;
        let sr = 48_000.0;
        let freq = 440.0;
        let phase_a = 1.0;
        let phase_b = 2.0;
        let engine_a: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr + phase_a).sin())
            .collect();
        let engine_b: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr + phase_b).sin())
            .collect();
        let result = spectral_intersection(&[&engine_a, &engine_b], false);
        let expected_phase = 0.5 * (phase_a + phase_b);
        let actual_phase = phase_at_frequency(&result, freq, sr);
        assert!(
            wrap_phase_error(actual_phase, expected_phase) < 0.15,
            "Consensus phase should follow the circular mean below original Nyquist \
             (actual={actual_phase:.4}, expected={expected_phase:.4})",
        );
    }

    #[test]
    fn circular_mean_attenuates_bins_with_phase_disagreement() {
        // Opposed low-frequency phases should attenuate the bin just as strongly
        // as opposed high-frequency phases.
        let n = 8192;
        let sr = 48_000.0;
        let freq = 440.0;
        let engine_a: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let engine_b: Vec<f64> = engine_a.iter().map(|&x| -x).collect();

        let result = spectral_intersection(&[&engine_a, &engine_b], false);

        let rms_result = rms(&result);
        let rms_engine = rms(&engine_a);
        assert!(
            rms_result < rms_engine * 0.15,
            "Opposite-phase engines should be heavily attenuated across the spectrum \
             (result RMS {rms_result:.6} vs engine RMS {rms_engine:.6})",
        );
    }

    #[test]
    fn circular_mean_preserves_bins_with_phase_agreement() {
        // Matching low-frequency phases should preserve shared energy as well.
        let n = 8192;
        let sr = 48_000.0;
        let freq = 440.0;
        let engine_a: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let engine_b = engine_a.clone();

        let result = spectral_intersection(&[&engine_a, &engine_b], false);

        let rms_result = rms(&result);
        let rms_engine = rms(&engine_a);
        assert!(
            rms_result > rms_engine * 0.5,
            "Same-phase engines should preserve most shared energy across the spectrum \
             (result RMS {rms_result:.6} vs engine RMS {rms_engine:.6})",
        );
    }

    #[test]
    fn frequency_blend_locks_phase_at_low_frequency_to_source() {
        // At 100 Hz with source Nyquist 4 kHz the linear weight
        // w_src = 1 - 100/4000 = 0.975, so source phase should dominate
        // the candidate's opposing phase almost entirely. Source is
        // provided at the output rate because the caller (remaster.rs)
        // resamples before invoking the blend.
        let source_rate = 8_000;
        let output_rate = 48_000;
        let output_frames = 24_576;
        let low_freq = 100.0;
        let source = tone(output_rate, output_frames, low_freq, 1.0, 0.0);
        let candidate = mix(&[
            tone(
                output_rate,
                output_frames,
                low_freq,
                1.0,
                std::f64::consts::PI,
            ),
            tone(output_rate, output_frames, 12_000.0, 0.4, 0.0),
        ]);

        let blended =
            apply_source_frequency_blend(&candidate, &source, source_rate, output_rate, false);

        let blended_phase = phase_at_frequency(&blended, low_freq, output_rate as f64);
        assert!(
            wrap_phase_error(blended_phase, 0.0) < 0.15,
            "Low-frequency phase should lock to source (blended={blended_phase:.4}, expected ~0)",
        );
    }

    #[test]
    fn frequency_blend_preserves_content_above_source_nyquist() {
        // Source Nyquist is 4 kHz; a 12 kHz candidate component must pass
        // through the blend untouched because the source has no information
        // at that frequency.
        let source_rate = 8_000;
        let output_rate = 48_000;
        let output_frames = 24_576;
        let source = tone(output_rate, output_frames, 440.0, 1.0, 0.0);
        let candidate = mix(&[
            tone(output_rate, output_frames, 440.0, 1.0, 0.0),
            tone(output_rate, output_frames, 12_000.0, 0.6, 0.7),
        ]);

        let blended =
            apply_source_frequency_blend(&candidate, &source, source_rate, output_rate, false);

        let before = component_amplitude(&candidate, 12_000.0, output_rate);
        let after = component_amplitude(&blended, 12_000.0, output_rate);
        let relative_change = (after - before).abs() / before.max(1e-12);
        assert!(
            relative_change < 0.05,
            "Content above source Nyquist must pass through unchanged \
             (before={before:.4}, after={after:.4}, change={relative_change:.4})",
        );

        let cand_phase = phase_at_frequency(&candidate, 12_000.0, output_rate as f64);
        let blend_phase = phase_at_frequency(&blended, 12_000.0, output_rate as f64);
        assert!(
            wrap_phase_error(blend_phase, cand_phase) < 0.1,
            "Candidate phase above source Nyquist must be preserved \
             (blend={blend_phase:.4}, cand={cand_phase:.4})",
        );
    }

    #[test]
    fn frequency_blend_uses_linear_weights_at_mid_band() {
        // At f = source_nyquist / 2 = 2 kHz, linear weights give
        // (w_src, w_cand) = (0.5, 0.5). When only the source carries content
        // at that bin, the blended amplitude should equal 0.5 * source_amp.
        let source_rate = 8_000;
        let output_rate = 48_000;
        let output_frames = 24_576;
        let mid_freq = 2_000.0;
        let source_amp = 1.0;
        let source = tone(output_rate, output_frames, mid_freq, source_amp, 0.0);
        // Candidate has only passthrough content far above source Nyquist so
        // the mid-band blend reads only from the source side.
        let candidate = tone(output_rate, output_frames, 12_000.0, 0.3, 0.0);

        let blended =
            apply_source_frequency_blend(&candidate, &source, source_rate, output_rate, false);

        let expected = source_amp * 0.5;
        let actual = component_amplitude(&blended, mid_freq, output_rate);
        let relative_error = (actual - expected).abs() / expected;
        assert!(
            relative_error < 0.15,
            "Linear mid-band amplitude should be 0.5 * source \
             (expected={expected:.4}, actual={actual:.4})",
        );
    }

    #[test]
    fn frequency_blend_looped_preserves_wrap_continuity() {
        // A loop-continuous low-frequency tone stays loop-continuous after
        // the blend — the wrap-padded STFT path must not introduce seam
        // artifacts at the loop boundary. Use only low-frequency content so
        // adjacent-sample deltas stay small and the continuity check is
        // meaningful (high-frequency tones have large intrinsic per-sample
        // swings that would swamp the test).
        let source_rate = 8_000;
        let output_rate = 48_000;
        let output_frames = 24_576;
        let cycles = 225.0;
        let loop_freq = cycles * output_rate as f64 / output_frames as f64;
        let source = tone(output_rate, output_frames, loop_freq, 1.0, 0.0);
        let candidate = tone(output_rate, output_frames, loop_freq, 1.0, 0.0);

        let blended =
            apply_source_frequency_blend(&candidate, &source, source_rate, output_rate, true);

        let gap = (blended[blended.len() - 1] - blended[0]).abs();
        let peak = blended.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let relative_gap = if peak > 0.0 { gap / peak } else { 0.0 };
        assert!(
            relative_gap < 0.1,
            "Looped spectral blend should preserve wrap continuity \
             (relative_gap={relative_gap:.4}, gap={gap:.6}, peak={peak:.6})",
        );
    }

    #[test]
    fn frequency_blend_output_stays_real_after_complex_mix() {
        // A complex-valued blend has to restore conjugate symmetry; if it
        // doesn't, ISTFT output would carry an imaginary tail that we drop.
        // Check that the blended signal has no pathological values and that
        // the low-frequency source content survives the round-trip at close
        // to full amplitude.
        let source_rate = 8_000;
        let output_rate = 48_000;
        let output_frames = 24_576;
        let source_amp = 0.5;
        let source = tone(output_rate, output_frames, 440.0, source_amp, 0.3);
        let candidate = mix(&[
            tone(output_rate, output_frames, 440.0, source_amp, 0.3),
            tone(output_rate, output_frames, 11_000.0, 0.4, 1.1),
        ]);

        let blended =
            apply_source_frequency_blend(&candidate, &source, source_rate, output_rate, false);

        for (i, &sample) in blended.iter().enumerate() {
            assert!(sample.is_finite(), "Sample {i} is not finite: {sample}");
            assert!(sample.abs() < 5.0, "Sample {i} magnitude blew up: {sample}");
        }

        let low_out = component_amplitude(&blended, 440.0, output_rate);
        let hi_out = component_amplitude(&blended, 11_000.0, output_rate);
        // At 440 Hz both source and candidate carry the same tone in-phase,
        // so linear weights simply recover the input amplitude (no boost).
        // At 11 kHz (above source Nyquist) the candidate passes through.
        assert!(
            (low_out - source_amp).abs() < 0.1,
            "Low-frequency in-phase blend should recover source amp (got {low_out:.4})",
        );
        assert!(
            (hi_out - 0.4).abs() < 0.05,
            "High-frequency passthrough should match candidate amp (got {hi_out:.4})",
        );
    }

    fn phase_at_frequency(signal: &[f64], freq_hz: f64, sample_rate: f64) -> f64 {
        let mut sin_dot = 0.0;
        let mut cos_dot = 0.0;
        for (idx, &sample) in signal.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * freq_hz * idx as f64 / sample_rate;
            sin_dot += sample * phase.sin();
            cos_dot += sample * phase.cos();
        }
        cos_dot.atan2(sin_dot)
    }

    fn wrap_phase_error(actual: f64, expected: f64) -> f64 {
        let diff = (actual - expected + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        diff.abs()
    }

    fn pure_sine(frames: usize, freq_hz: f64, sample_rate_hz: u32) -> Vec<f64> {
        let omega = 2.0 * std::f64::consts::PI * freq_hz / sample_rate_hz as f64;
        (0..frames).map(|i| (omega * i as f64).sin()).collect()
    }

    #[test]
    fn hf_energy_envelope_isolates_above_cutoff() {
        let rate = 48_000u32;
        let frames = 8192usize;
        // Signal well above the cutoff: non-zero HF energy.
        let cutoff = 4000.0;
        let above = pure_sine(frames, 8000.0, rate);
        let env_above = hf_energy_envelope(&above, rate, cutoff);
        assert!(!env_above.is_empty(), "expected non-empty envelope");
        let total_above: f64 = env_above.iter().sum();
        assert!(
            total_above > 1.0,
            "HF energy above cutoff should be non-trivial: {total_above}"
        );

        // Signal below the cutoff: at least 3 orders of magnitude less HF energy
        // than above-cutoff (hann leakage is always finite).
        let below = pure_sine(frames, 1000.0, rate);
        let env_below = hf_energy_envelope(&below, rate, cutoff);
        let total_below: f64 = env_below.iter().sum();
        assert!(
            total_below < total_above * 1.0e-3,
            "HF energy below cutoff should be << above: below={total_below} above={total_above}"
        );
    }

    #[test]
    fn hf_energy_envelope_detects_localized_burst() {
        let rate = 48_000u32;
        let frames = 16_384usize;
        let cutoff = 4000.0;
        let mut signal = pure_sine(frames, 8000.0, rate);
        // Quadruple the amplitude over a short burst in the middle.
        let burst_start = frames / 2;
        let burst_len = 256;
        for sample in signal.iter_mut().skip(burst_start).take(burst_len) {
            *sample *= 4.0;
        }
        let env = hf_energy_envelope(&signal, rate, cutoff);
        let avg: f64 = env.iter().sum::<f64>() / env.len() as f64;
        let max = env.iter().copied().fold(0.0f64, f64::max);
        assert!(
            max / avg > 3.0,
            "burst should produce peak/avg > 3: peak={max} avg={avg}"
        );
    }

    #[test]
    fn stft_frame_for_sample_is_monotonic() {
        let rate = 48_000u32;
        let a = stft_frame_for_sample(0, rate);
        let b = stft_frame_for_sample(1024, rate);
        let c = stft_frame_for_sample(8192, rate);
        assert!(a <= b && b <= c);
    }

    #[test]
    fn cross_correlation_peak_at_zero_lag() {
        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let corr = fft_cross_correlation(&signal, &signal);
        assert!(!corr.is_empty());
        let peak_lag = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_lag, 0, "self-correlation should peak at lag 0");
    }

    #[test]
    fn cross_correlation_peak_at_known_offset() {
        let template: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let offset = 100;
        let mut signal = vec![0.0; 300];
        for (i, &v) in template.iter().enumerate() {
            signal[offset + i] = v;
        }
        let corr = fft_cross_correlation(&template, &signal);
        let peak_lag = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            peak_lag, offset,
            "peak should appear at the insertion offset"
        );
    }

    #[test]
    fn cross_correlation_reversed_template() {
        let template: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let reversed: Vec<f64> = template.iter().copied().rev().collect();
        let offset = 80;
        let mut signal = vec![0.0; 250];
        for (i, &v) in reversed.iter().enumerate() {
            signal[offset + i] = v;
        }
        let corr = fft_cross_correlation(&reversed, &signal);
        let peak_lag = corr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            peak_lag, offset,
            "reversed template should match reversed segment"
        );
    }

    #[test]
    fn cross_correlation_empty_inputs() {
        assert!(fft_cross_correlation(&[], &[1.0, 2.0]).is_empty());
        assert!(fft_cross_correlation(&[1.0], &[]).is_empty());
    }
}
