// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

// f32 scalar functions are unused in production (pipeline is fully f64) but retained
// as ground-truth reference implementations for SIMD dispatch correctness tests.
#![allow(clippy::needless_return, dead_code)]

use pulp::{Simd, WithSimd};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use pulp::Arch;

pub(crate) const I32_PCM_SCALE: f32 = 2_147_483_647.0;
pub(crate) const I32_PCM_SCALE_F64: f64 = 2_147_483_647.0;

#[allow(dead_code)]
#[inline]
pub fn mix_to_mono_scalar(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    let frames = data.len() / channels;
    (0..frames)
        .map(|i| {
            let sum: f32 = (0..channels).map(|ch| data[i * channels + ch]).sum();
            sum / channels as f32
        })
        .collect()
}

#[cfg(test)]
#[inline]
pub fn mix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels != 2 {
        return mix_to_mono_scalar(data, channels);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(StereoMixToMono { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        mix_to_mono_scalar(data, channels)
    }
}

#[allow(dead_code)]
#[inline]
pub fn pearson_correlation_scalar(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }

    let a = &a[..n as usize];
    let b = &b[..n as usize];
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}

/// NOTE on determinism: the SIMD path uses FMA with unrolled per-lane
/// accumulators, so results differ from the scalar reference — and across
/// ISA tiers (AVX-512 / AVX2 / scalar fallback) — by ~1e-12..1e-10. This
/// score is compared against the 0.9 usable-score floor downstream, so a
/// sample scoring within ~1e-10 of the floor can pass on one machine and
/// fail on another. Within one machine, results are deterministic.
#[inline]
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(PearsonCorrelation { a, b });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        pearson_correlation_scalar(a, b)
    }
}

#[allow(dead_code)]
#[inline]
pub fn rms_f32_scalar(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let sum = data
        .iter()
        .map(|&sample| (sample as f64) * (sample as f64))
        .sum::<f64>();
    (sum / data.len() as f64).sqrt() as f32
}

#[allow(dead_code)]
#[inline]
pub fn rms_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(RmsF32 { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        rms_f32_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn peak_abs_f32_scalar(data: &[f32]) -> f32 {
    data.iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max)
}

#[allow(dead_code)]
#[inline]
pub fn peak_abs_f32(data: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(PeakAbsF32 { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        peak_abs_f32_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_in_place_scalar(data: &mut [f32], gain: f32) {
    for sample in data {
        *sample *= gain;
    }
}

#[inline]
pub fn scale_in_place(data: &mut [f32], gain: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Arch::new().dispatch(ScaleInPlace { data, gain });
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        scale_in_place_scalar(data, gain);
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_and_clamp_in_place_scalar(data: &mut [f32], gain: f32, min: f32, max: f32) {
    for sample in data {
        *sample = (*sample * gain).clamp(min, max);
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_and_clamp_in_place(data: &mut [f32], gain: f32, min: f32, max: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Arch::new().dispatch(ScaleAndClampInPlace {
            data,
            gain,
            min,
            max,
        });
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        scale_and_clamp_in_place_scalar(data, gain, min, max);
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i16_scalar(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&sample| (sample * 32_767.0) as i16)
        .collect()
}

#[cfg(test)]
#[inline]
pub fn scale_to_i16(samples: &[f32]) -> Vec<i16> {
    let mut scaled = samples.to_vec();
    scale_in_place(&mut scaled, 32_767.0);
    scaled.into_iter().map(|sample| sample as i16).collect()
}


/// Quantize one sample to signed 24-bit with round-half-away and clamping.
/// Out-of-range input must CLAMP, not wrap: taking the low 3 bytes of an
/// unclamped i32 polarity-inverts anything past full scale (e.g. +1.5 used
/// to decode as -0.5). Truncation is also replaced by rounding to match the
/// FLAC writer's convention. NaN quantizes to 0.
#[inline]
fn quantize_i24(scaled: f64) -> [u8; 4] {
    const MAX_24: f64 = 8_388_607.0;
    const MIN_24: f64 = -8_388_608.0;
    // `clamp` propagates NaN and `NaN as i32` is 0 — NaN becomes silence.
    let v = scaled.round().clamp(MIN_24, MAX_24);
    (v as i32).to_le_bytes()
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i24le_bytes_scalar(samples: &[f32]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 3);
    for &sample in samples {
        // f32 multiply (not f64) so the product bit-matches the dispatched
        // path, which scales via the f32 SIMD kernel before quantizing.
        let bytes = quantize_i24((sample * 8_388_607.0f32) as f64);
        pcm_bytes.extend_from_slice(&bytes[..3]);
    }
    pcm_bytes
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i24le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut scaled = samples.to_vec();
    scale_in_place(&mut scaled, 8_388_607.0);

    let mut pcm_bytes = Vec::with_capacity(scaled.len() * 3);
    for sample in scaled {
        let bytes = quantize_i24(sample as f64);
        pcm_bytes.extend_from_slice(&bytes[..3]);
    }
    pcm_bytes
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i32le_bytes_scalar(samples: &[f32]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 4);
    for &sample in samples {
        pcm_bytes.extend_from_slice(&((sample * I32_PCM_SCALE) as i32).to_le_bytes());
    }
    pcm_bytes
}

#[cfg(test)]
#[inline]
pub fn scale_to_i32le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut scaled = samples.to_vec();
    scale_in_place(&mut scaled, I32_PCM_SCALE);

    let mut pcm_bytes = Vec::with_capacity(scaled.len() * 4);
    for sample in scaled {
        pcm_bytes.extend_from_slice(&(sample as i32).to_le_bytes());
    }
    pcm_bytes
}

// ── New kernels: sum_f32, subtract_in_place, scale_from_i24le_bytes,
//    deinterleave_stereo, interleave_stereo ──

#[allow(dead_code)]
#[inline]
pub fn sum_f32_scalar(data: &[f32]) -> f64 {
    data.iter().map(|&s| s as f64).sum()
}

#[allow(dead_code)]
#[inline]
pub fn sum_f32(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(SumF32 { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        sum_f32_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn subtract_in_place_scalar(data: &mut [f32], value: f32) {
    for sample in data {
        *sample -= value;
    }
}

#[allow(dead_code)]
#[inline]
pub fn subtract_in_place(data: &mut [f32], value: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Arch::new().dispatch(SubtractInPlace { data, value });
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        subtract_in_place_scalar(data, value);
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_from_i24le_bytes_scalar(pcm: &[u8]) -> Vec<f32> {
    let scale = 1.0 / 8_388_608.0f32;
    pcm.chunks_exact(3)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..3].copy_from_slice(chunk);
            if bytes[2] & 0x80 != 0 {
                bytes[3] = 0xFF;
            }
            i32::from_le_bytes(bytes) as f32 * scale
        })
        .collect()
}

#[allow(dead_code)]
#[inline]
pub fn scale_from_i24le_bytes(pcm: &[u8]) -> Vec<f32> {
    let n = pcm.len() / 3;
    let mut data = Vec::with_capacity(n);
    for chunk in pcm.chunks_exact(3) {
        let mut bytes = [0u8; 4];
        bytes[..3].copy_from_slice(chunk);
        if bytes[2] & 0x80 != 0 {
            bytes[3] = 0xFF;
        }
        data.push(i32::from_le_bytes(bytes) as f32);
    }
    scale_in_place(&mut data, 1.0 / 8_388_608.0);
    data
}

#[allow(dead_code)]
#[inline]
pub fn deinterleave_stereo_scalar(data: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let frames = data.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for pair in data.chunks_exact(2) {
        left.push(pair[0]);
        right.push(pair[1]);
    }
    (left, right)
}

#[allow(dead_code)]
#[inline]
pub fn deinterleave_stereo(data: &[f32]) -> (Vec<f32>, Vec<f32>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(DeinterleaveStereo { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        deinterleave_stereo_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn interleave_stereo_scalar(left: &[f32], right: &[f32]) -> Vec<f32> {
    let frames = left.len().min(right.len());
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        out.push(left[i]);
        out.push(right[i]);
    }
    out
}

#[allow(dead_code)]
#[inline]
pub fn interleave_stereo(left: &[f32], right: &[f32]) -> Vec<f32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(InterleaveStereo { left, right });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        interleave_stereo_scalar(left, right)
    }
}
#[cfg(test)]
struct StereoMixToMono<'a> {
    data: &'a [f32],
}

#[cfg(test)]
impl WithSimd for StereoMixToMono<'_> {
    type Output = Vec<f32>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        let frames = data.len() / 2;
        let mut output = vec![0.0f32; frames];
        let half = simd.splat_f32s(0.5);
        let lanes = S::F32_LANES;
        let simd_frames = frames / lanes;
        let mut left = vec![0.0f32; lanes];
        let mut right = vec![0.0f32; lanes];

        for block in 0..simd_frames {
            let start = block * lanes * 2;
            for (lane, frame) in data[start..start + lanes * 2].chunks_exact(2).enumerate() {
                left[lane] = frame[0];
                right[lane] = frame[1];
            }

            let mixed = simd.mul_f32s(
                simd.add_f32s(
                    simd.partial_load_f32s(&left),
                    simd.partial_load_f32s(&right),
                ),
                half,
            );
            simd.partial_store_f32s(&mut output[block * lanes..(block + 1) * lanes], mixed);
        }

        let processed_frames = simd_frames * lanes;
        for (frame, out) in data[processed_frames * 2..]
            .chunks_exact(2)
            .zip(output[processed_frames..].iter_mut())
        {
            *out = (frame[0] + frame[1]) * 0.5;
        }

        output
    }
}

struct PearsonCorrelation<'a> {
    a: &'a [f64],
    b: &'a [f64],
}

impl WithSimd for PearsonCorrelation<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let len = self.a.len().min(self.b.len());
        if len < 2 {
            return 0.0;
        }

        let a = &self.a[..len];
        let b = &self.b[..len];
        let mean_a = sum_f64_with_simd(simd, a) / len as f64;
        let mean_b = sum_f64_with_simd(simd, b) / len as f64;

        let (a_head, a_tail) = S::as_simd_f64s(a);
        let (b_head, b_tail) = S::as_simd_f64s(b);
        let mean_a_vec = simd.splat_f64s(mean_a);
        let mean_b_vec = simd.splat_f64s(mean_b);

        let mut cov0 = simd.splat_f64s(0.0);
        let mut cov1 = simd.splat_f64s(0.0);
        let mut cov2 = simd.splat_f64s(0.0);
        let mut cov3 = simd.splat_f64s(0.0);
        let mut var_a0 = simd.splat_f64s(0.0);
        let mut var_a1 = simd.splat_f64s(0.0);
        let mut var_a2 = simd.splat_f64s(0.0);
        let mut var_a3 = simd.splat_f64s(0.0);
        let mut var_b0 = simd.splat_f64s(0.0);
        let mut var_b1 = simd.splat_f64s(0.0);
        let mut var_b2 = simd.splat_f64s(0.0);
        let mut var_b3 = simd.splat_f64s(0.0);

        let (a4, a1) = pulp::as_arrays::<4, _>(a_head);
        let (b4, b1) = pulp::as_arrays::<4, _>(b_head);

        for (&[a0, a1v, a2, a3], &[b0, b1v, b2, b3]) in a4.iter().zip(b4.iter()) {
            let da0 = simd.sub_f64s(a0, mean_a_vec);
            let db0 = simd.sub_f64s(b0, mean_b_vec);
            let da1 = simd.sub_f64s(a1v, mean_a_vec);
            let db1 = simd.sub_f64s(b1v, mean_b_vec);
            let da2 = simd.sub_f64s(a2, mean_a_vec);
            let db2 = simd.sub_f64s(b2, mean_b_vec);
            let da3 = simd.sub_f64s(a3, mean_a_vec);
            let db3 = simd.sub_f64s(b3, mean_b_vec);

            cov0 = simd.mul_add_f64s(da0, db0, cov0);
            cov1 = simd.mul_add_f64s(da1, db1, cov1);
            cov2 = simd.mul_add_f64s(da2, db2, cov2);
            cov3 = simd.mul_add_f64s(da3, db3, cov3);
            var_a0 = simd.mul_add_f64s(da0, da0, var_a0);
            var_a1 = simd.mul_add_f64s(da1, da1, var_a1);
            var_a2 = simd.mul_add_f64s(da2, da2, var_a2);
            var_a3 = simd.mul_add_f64s(da3, da3, var_a3);
            var_b0 = simd.mul_add_f64s(db0, db0, var_b0);
            var_b1 = simd.mul_add_f64s(db1, db1, var_b1);
            var_b2 = simd.mul_add_f64s(db2, db2, var_b2);
            var_b3 = simd.mul_add_f64s(db3, db3, var_b3);
        }

        for (&av, &bv) in a1.iter().zip(b1.iter()) {
            let da = simd.sub_f64s(av, mean_a_vec);
            let db = simd.sub_f64s(bv, mean_b_vec);
            cov0 = simd.mul_add_f64s(da, db, cov0);
            var_a0 = simd.mul_add_f64s(da, da, var_a0);
            var_b0 = simd.mul_add_f64s(db, db, var_b0);
        }

        cov0 = simd.add_f64s(cov0, cov1);
        cov2 = simd.add_f64s(cov2, cov3);
        var_a0 = simd.add_f64s(var_a0, var_a1);
        var_a2 = simd.add_f64s(var_a2, var_a3);
        var_b0 = simd.add_f64s(var_b0, var_b1);
        var_b2 = simd.add_f64s(var_b2, var_b3);

        let mut cov = simd.reduce_sum_f64s(simd.add_f64s(cov0, cov2));
        let mut var_a = simd.reduce_sum_f64s(simd.add_f64s(var_a0, var_a2));
        let mut var_b = simd.reduce_sum_f64s(simd.add_f64s(var_b0, var_b2));

        for (&ai, &bi) in a_tail.iter().zip(b_tail.iter()) {
            let da = ai - mean_a;
            let db = bi - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        cov / denom
    }
}

#[allow(dead_code)]
struct RmsF32<'a> {
    data: &'a [f32],
}

impl WithSimd for RmsF32<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        if data.is_empty() {
            return 0.0;
        }

        let (head, tail) = S::as_simd_f32s(data);
        let (head4, head1) = pulp::as_arrays::<4, _>(head);

        // Reduce to an f64 running sum per block so f32 accumulator error
        // stays bounded for arbitrarily long buffers — the scalar reference
        // accumulates squares in f64 throughout, and multi-minute buffers
        // exceeded the parity tolerance with pure-f32 accumulation.
        let mut sum = 0.0f64;
        for block in head4.chunks(1024) {
            let mut acc0 = simd.splat_f32s(0.0);
            let mut acc1 = simd.splat_f32s(0.0);
            let mut acc2 = simd.splat_f32s(0.0);
            let mut acc3 = simd.splat_f32s(0.0);
            for &[x0, x1, x2, x3] in block {
                acc0 = simd.mul_add_f32s(x0, x0, acc0);
                acc1 = simd.mul_add_f32s(x1, x1, acc1);
                acc2 = simd.mul_add_f32s(x2, x2, acc2);
                acc3 = simd.mul_add_f32s(x3, x3, acc3);
            }
            acc0 = simd.add_f32s(acc0, acc1);
            acc2 = simd.add_f32s(acc2, acc3);
            sum += simd.reduce_sum_f32s(simd.add_f32s(acc0, acc2)) as f64;
        }
        {
            let mut acc = simd.splat_f32s(0.0);
            for &x in head1 {
                acc = simd.mul_add_f32s(x, x, acc);
            }
            sum += simd.reduce_sum_f32s(acc) as f64;
        }
        for &sample in tail {
            sum += (sample as f64) * (sample as f64);
        }

        (sum / data.len() as f64).sqrt() as f32
    }
}

#[allow(dead_code)]
struct PeakAbsF32<'a> {
    data: &'a [f32],
}

impl WithSimd for PeakAbsF32<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        let (head, tail) = S::as_simd_f32s(data);
        let mut peak = simd.splat_f32s(0.0);

        for &values in head {
            peak = simd.max_f32s(peak, simd.abs_f32s(values));
        }

        let mut reduced = if head.is_empty() {
            0.0
        } else {
            simd.reduce_max_f32s(peak)
        };
        for &sample in tail {
            reduced = reduced.max(sample.abs());
        }
        reduced
    }
}

struct ScaleInPlace<'a> {
    data: &'a mut [f32],
    gain: f32,
}

impl WithSimd for ScaleInPlace<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data, gain } = self;
        let gain_vec = simd.splat_f32s(gain);
        let (head, tail) = S::as_mut_simd_f32s(data);

        for values in head {
            *values = simd.mul_f32s(*values, gain_vec);
        }

        for sample in tail {
            *sample *= gain;
        }
    }
}

#[allow(dead_code)]
struct ScaleAndClampInPlace<'a> {
    data: &'a mut [f32],
    gain: f32,
    min: f32,
    max: f32,
}

impl WithSimd for ScaleAndClampInPlace<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            data,
            gain,
            min,
            max,
        } = self;
        let gain_vec = simd.splat_f32s(gain);
        let min_vec = simd.splat_f32s(min);
        let max_vec = simd.splat_f32s(max);
        let (head, tail) = S::as_mut_simd_f32s(data);

        for values in head {
            let scaled = simd.mul_f32s(*values, gain_vec);
            *values = simd.max_f32s(min_vec, simd.min_f32s(max_vec, scaled));
        }

        for sample in tail {
            *sample = (*sample * gain).clamp(min, max);
        }
    }
}

#[inline(always)]
fn sum_f64_with_simd<S: Simd>(simd: S, data: &[f64]) -> f64 {
    let (head, tail) = S::as_simd_f64s(data);
    let (head4, head1) = pulp::as_arrays::<4, _>(head);
    let mut acc0 = simd.splat_f64s(0.0);
    let mut acc1 = simd.splat_f64s(0.0);
    let mut acc2 = simd.splat_f64s(0.0);
    let mut acc3 = simd.splat_f64s(0.0);

    for &[x0, x1, x2, x3] in head4 {
        acc0 = simd.add_f64s(acc0, x0);
        acc1 = simd.add_f64s(acc1, x1);
        acc2 = simd.add_f64s(acc2, x2);
        acc3 = simd.add_f64s(acc3, x3);
    }

    for &x in head1 {
        acc0 = simd.add_f64s(acc0, x);
    }

    acc0 = simd.add_f64s(acc0, acc1);
    acc2 = simd.add_f64s(acc2, acc3);
    let mut sum = simd.reduce_sum_f64s(simd.add_f64s(acc0, acc2));
    for &value in tail {
        sum += value;
    }
    sum
}

#[allow(dead_code)]
struct SumF32<'a> {
    data: &'a [f32],
}

impl WithSimd for SumF32<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        const CHUNK_SIZE: usize = 256;

        let Self { data } = self;
        if data.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut widened = [0.0f64; CHUNK_SIZE];
        for chunk in data.chunks(CHUNK_SIZE) {
            for (dst, &src) in widened[..chunk.len()].iter_mut().zip(chunk.iter()) {
                *dst = src as f64;
            }
            sum += sum_f64_with_simd(simd, &widened[..chunk.len()]);
        }
        sum
    }
}

#[allow(dead_code)]
struct SubtractInPlace<'a> {
    data: &'a mut [f32],
    value: f32,
}

impl WithSimd for SubtractInPlace<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data, value } = self;
        let val_vec = simd.splat_f32s(value);
        let (head, tail) = S::as_mut_simd_f32s(data);

        for values in head {
            *values = simd.sub_f32s(*values, val_vec);
        }

        for sample in tail {
            *sample -= value;
        }
    }
}

#[allow(dead_code)]
struct DeinterleaveStereo<'a> {
    data: &'a [f32],
}

impl WithSimd for DeinterleaveStereo<'_> {
    type Output = (Vec<f32>, Vec<f32>);

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        let frames = data.len() / 2;
        let mut left = vec![0.0f32; frames];
        let mut right = vec![0.0f32; frames];
        let lanes = S::F32_LANES;
        let simd_frames = frames / lanes;
        let mut left_buf = vec![0.0f32; lanes];
        let mut right_buf = vec![0.0f32; lanes];

        for block in 0..simd_frames {
            let start = block * lanes * 2;
            for (lane, frame) in data[start..start + lanes * 2].chunks_exact(2).enumerate() {
                left_buf[lane] = frame[0];
                right_buf[lane] = frame[1];
            }

            simd.partial_store_f32s(
                &mut left[block * lanes..(block + 1) * lanes],
                simd.partial_load_f32s(&left_buf),
            );
            simd.partial_store_f32s(
                &mut right[block * lanes..(block + 1) * lanes],
                simd.partial_load_f32s(&right_buf),
            );
        }

        let processed = simd_frames * lanes;
        for (i, pair) in data[processed * 2..].chunks_exact(2).enumerate() {
            left[processed + i] = pair[0];
            right[processed + i] = pair[1];
        }

        (left, right)
    }
}

#[allow(dead_code)]
struct InterleaveStereo<'a> {
    left: &'a [f32],
    right: &'a [f32],
}

impl WithSimd for InterleaveStereo<'_> {
    type Output = Vec<f32>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { left, right } = self;
        let frames = left.len().min(right.len());
        let mut out = vec![0.0f32; frames * 2];
        let lanes = S::F32_LANES;
        let simd_frames = frames / lanes;
        let mut left_buf = vec![0.0f32; lanes];
        let mut right_buf = vec![0.0f32; lanes];

        for block in 0..simd_frames {
            let offset = block * lanes;
            simd.partial_store_f32s(
                &mut left_buf,
                simd.partial_load_f32s(&left[offset..offset + lanes]),
            );
            simd.partial_store_f32s(
                &mut right_buf,
                simd.partial_load_f32s(&right[offset..offset + lanes]),
            );

            let out_start = block * lanes * 2;
            for lane in 0..lanes {
                out[out_start + lane * 2] = left_buf[lane];
                out[out_start + lane * 2 + 1] = right_buf[lane];
            }
        }

        let processed = simd_frames * lanes;
        for i in processed..frames {
            out[i * 2] = left[i];
            out[i * 2 + 1] = right[i];
        }

        out
    }
}

// ── f64 sample-data pipeline variants ──
// These mirror the f32 functions above but operate on f64 data,
// used by the remaster pipeline where sample data is now f64.

#[allow(dead_code)]
#[inline]
pub fn rms_f64_scalar(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().map(|&s| s * s).sum();
    (sum / data.len() as f64).sqrt()
}

#[inline]
pub fn rms_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(RmsF64 { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        rms_f64_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn peak_abs_f64_scalar(data: &[f64]) -> f64 {
    data.iter()
        .map(|sample| sample.abs())
        .fold(0.0f64, f64::max)
}

#[inline]
pub fn peak_abs_f64(data: &[f64]) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(PeakAbsF64 { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        peak_abs_f64_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_in_place_f64_scalar(data: &mut [f64], gain: f64) {
    for sample in data {
        *sample *= gain;
    }
}

#[inline]
pub fn scale_in_place_f64(data: &mut [f64], gain: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Arch::new().dispatch(ScaleInPlaceF64 { data, gain });
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        scale_in_place_f64_scalar(data, gain);
    }
}

#[allow(dead_code)]
#[inline]
pub fn sum_f64_dispatch_scalar(data: &[f64]) -> f64 {
    data.iter().sum()
}

#[inline]
pub fn sum_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(SumF64Dispatch { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        sum_f64_dispatch_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn subtract_in_place_f64_scalar(data: &mut [f64], value: f64) {
    for sample in data {
        *sample -= value;
    }
}

#[inline]
pub fn subtract_in_place_f64(data: &mut [f64], value: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Arch::new().dispatch(SubtractInPlaceF64 { data, value });
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        subtract_in_place_f64_scalar(data, value);
    }
}

#[allow(dead_code)]
#[inline]
pub fn add_in_place_f64_scalar(data: &mut [f64], other: &[f64]) {
    for (sample, addend) in data.iter_mut().zip(other.iter().copied()) {
        *sample += addend;
    }
}

#[inline]
#[allow(dead_code)]
pub fn add_in_place_f64(data: &mut [f64], other: &[f64]) {
    let len = data.len().min(other.len());
    if len == 0 {
        return;
    }
    let (data, other) = (&mut data[..len], &other[..len]);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Arch::new().dispatch(AddInPlaceF64 { data, other });
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        add_in_place_f64_scalar(data, other);
    }
}

#[allow(dead_code)]
#[inline]
pub fn deinterleave_stereo_f64_scalar(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let frames = data.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for pair in data.chunks_exact(2) {
        left.push(pair[0]);
        right.push(pair[1]);
    }
    (left, right)
}

#[inline]
pub fn deinterleave_stereo_f64(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(DeinterleaveStereoF64 { data });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        deinterleave_stereo_f64_scalar(data)
    }
}

#[allow(dead_code)]
#[inline]
pub fn interleave_stereo_f64_scalar(left: &[f64], right: &[f64]) -> Vec<f64> {
    let frames = left.len().min(right.len());
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        out.push(left[i]);
        out.push(right[i]);
    }
    out
}

#[inline]
pub fn interleave_stereo_f64(left: &[f64], right: &[f64]) -> Vec<f64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return Arch::new().dispatch(InterleaveStereoF64 { left, right });
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        interleave_stereo_f64_scalar(left, right)
    }
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i24le_bytes_f64_scalar(samples: &[f64]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 3);
    for &sample in samples {
        let bytes = quantize_i24(sample * 8_388_607.0);
        pcm_bytes.extend_from_slice(&bytes[..3]);
    }
    pcm_bytes
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i24le_bytes_f64(samples: &[f64]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 3);
    for &sample in samples {
        let bytes = quantize_i24(sample * 8_388_607.0);
        pcm_bytes.extend_from_slice(&bytes[..3]);
    }
    pcm_bytes
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i32le_bytes_f64_scalar(samples: &[f64]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 4);
    for &sample in samples {
        pcm_bytes.extend_from_slice(&((sample * I32_PCM_SCALE_F64) as i32).to_le_bytes());
    }
    pcm_bytes
}

#[allow(dead_code)]
#[inline]
pub fn scale_to_i32le_bytes_f64(samples: &[f64]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 4);
    for &sample in samples {
        pcm_bytes.extend_from_slice(&((sample * I32_PCM_SCALE_F64) as i32).to_le_bytes());
    }
    pcm_bytes
}

#[allow(dead_code)]
#[inline]
pub fn scale_from_i24le_bytes_f64_scalar(pcm: &[u8]) -> Vec<f64> {
    let scale = 1.0 / 8_388_608.0f64;
    pcm.chunks_exact(3)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..3].copy_from_slice(chunk);
            if bytes[2] & 0x80 != 0 {
                bytes[3] = 0xFF;
            }
            i32::from_le_bytes(bytes) as f64 * scale
        })
        .collect()
}

#[inline]
pub fn scale_from_i24le_bytes_f64(pcm: &[u8]) -> Vec<f64> {
    let scale = 1.0 / 8_388_608.0f64;
    pcm.chunks_exact(3)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..3].copy_from_slice(chunk);
            if bytes[2] & 0x80 != 0 {
                bytes[3] = 0xFF;
            }
            i32::from_le_bytes(bytes) as f64 * scale
        })
        .collect()
}

#[allow(dead_code)]
#[inline]
pub fn mix_to_mono_f64_scalar(data: &[f64], channels: usize) -> Vec<f64> {
    if channels <= 1 {
        return data.to_vec();
    }
    let frames = data.len() / channels;
    (0..frames)
        .map(|i| {
            let sum: f64 = (0..channels).map(|ch| data[i * channels + ch]).sum();
            sum / channels as f64
        })
        .collect()
}

// ── f64 SIMD kernel structs ──

struct RmsF64<'a> {
    data: &'a [f64],
}

impl WithSimd for RmsF64<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        if data.is_empty() {
            return 0.0;
        }

        let (head, tail) = S::as_simd_f64s(data);
        let (head4, head1) = pulp::as_arrays::<4, _>(head);
        let mut acc0 = simd.splat_f64s(0.0);
        let mut acc1 = simd.splat_f64s(0.0);
        let mut acc2 = simd.splat_f64s(0.0);
        let mut acc3 = simd.splat_f64s(0.0);

        for &[x0, x1, x2, x3] in head4 {
            acc0 = simd.mul_add_f64s(x0, x0, acc0);
            acc1 = simd.mul_add_f64s(x1, x1, acc1);
            acc2 = simd.mul_add_f64s(x2, x2, acc2);
            acc3 = simd.mul_add_f64s(x3, x3, acc3);
        }

        for &x in head1 {
            acc0 = simd.mul_add_f64s(x, x, acc0);
        }

        acc0 = simd.add_f64s(acc0, acc1);
        acc2 = simd.add_f64s(acc2, acc3);
        let mut sum = simd.reduce_sum_f64s(simd.add_f64s(acc0, acc2));
        for &sample in tail {
            sum += sample * sample;
        }

        (sum / data.len() as f64).sqrt()
    }
}

struct PeakAbsF64<'a> {
    data: &'a [f64],
}

impl WithSimd for PeakAbsF64<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        let (head, tail) = S::as_simd_f64s(data);
        let mut peak = simd.splat_f64s(0.0);

        for &values in head {
            peak = simd.max_f64s(peak, simd.abs_f64s(values));
        }

        let mut reduced = if head.is_empty() {
            0.0
        } else {
            simd.reduce_max_f64s(peak)
        };
        for &sample in tail {
            reduced = reduced.max(sample.abs());
        }
        reduced
    }
}

struct ScaleInPlaceF64<'a> {
    data: &'a mut [f64],
    gain: f64,
}

impl WithSimd for ScaleInPlaceF64<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data, gain } = self;
        let gain_vec = simd.splat_f64s(gain);
        let (head, tail) = S::as_mut_simd_f64s(data);

        for values in head {
            *values = simd.mul_f64s(*values, gain_vec);
        }

        for sample in tail {
            *sample *= gain;
        }
    }
}

struct SumF64Dispatch<'a> {
    data: &'a [f64],
}

impl WithSimd for SumF64Dispatch<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        sum_f64_with_simd(simd, self.data)
    }
}

struct SubtractInPlaceF64<'a> {
    data: &'a mut [f64],
    value: f64,
}

impl WithSimd for SubtractInPlaceF64<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data, value } = self;
        let val_vec = simd.splat_f64s(value);
        let (head, tail) = S::as_mut_simd_f64s(data);

        for values in head {
            *values = simd.sub_f64s(*values, val_vec);
        }

        for sample in tail {
            *sample -= value;
        }
    }
}

struct AddInPlaceF64<'a> {
    data: &'a mut [f64],
    other: &'a [f64],
}

impl WithSimd for AddInPlaceF64<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data, other } = self;
        let len = data.len().min(other.len());
        let (data_head, data_tail) = S::as_mut_simd_f64s(&mut data[..len]);
        let (other_head, other_tail) = S::as_simd_f64s(&other[..len]);

        for (values, addend) in data_head.iter_mut().zip(other_head.iter()) {
            *values = simd.add_f64s(*values, *addend);
        }

        for (sample, addend) in data_tail.iter_mut().zip(other_tail.iter().copied()) {
            *sample += addend;
        }
    }
}

struct DeinterleaveStereoF64<'a> {
    data: &'a [f64],
}

impl WithSimd for DeinterleaveStereoF64<'_> {
    type Output = (Vec<f64>, Vec<f64>);

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { data } = self;
        let frames = data.len() / 2;
        let mut left = vec![0.0f64; frames];
        let mut right = vec![0.0f64; frames];
        let lanes = S::F64_LANES;
        let simd_frames = frames / lanes;
        let mut left_buf = vec![0.0f64; lanes];
        let mut right_buf = vec![0.0f64; lanes];

        for block in 0..simd_frames {
            let start = block * lanes * 2;
            for (lane, frame) in data[start..start + lanes * 2].chunks_exact(2).enumerate() {
                left_buf[lane] = frame[0];
                right_buf[lane] = frame[1];
            }

            simd.partial_store_f64s(
                &mut left[block * lanes..(block + 1) * lanes],
                simd.partial_load_f64s(&left_buf),
            );
            simd.partial_store_f64s(
                &mut right[block * lanes..(block + 1) * lanes],
                simd.partial_load_f64s(&right_buf),
            );
        }

        let processed = simd_frames * lanes;
        for (i, pair) in data[processed * 2..].chunks_exact(2).enumerate() {
            left[processed + i] = pair[0];
            right[processed + i] = pair[1];
        }

        (left, right)
    }
}

struct InterleaveStereoF64<'a> {
    left: &'a [f64],
    right: &'a [f64],
}

impl WithSimd for InterleaveStereoF64<'_> {
    type Output = Vec<f64>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { left, right } = self;
        let frames = left.len().min(right.len());
        let mut out = vec![0.0f64; frames * 2];
        let lanes = S::F64_LANES;
        let simd_frames = frames / lanes;
        let mut left_buf = vec![0.0f64; lanes];
        let mut right_buf = vec![0.0f64; lanes];

        for block in 0..simd_frames {
            let offset = block * lanes;
            simd.partial_store_f64s(
                &mut left_buf,
                simd.partial_load_f64s(&left[offset..offset + lanes]),
            );
            simd.partial_store_f64s(
                &mut right_buf,
                simd.partial_load_f64s(&right[offset..offset + lanes]),
            );

            let out_start = block * lanes * 2;
            for lane in 0..lanes {
                out[out_start + lane * 2] = left_buf[lane];
                out[out_start + lane * 2 + 1] = right_buf[lane];
            }
        }

        let processed = simd_frames * lanes;
        for i in processed..frames {
            out[i * 2] = left[i];
            out[i * 2 + 1] = right[i];
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REDUCTION_EPSILON: f64 = 1e-10;
    const F32_EPSILON: f32 = 1e-6;

    fn test_signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.13).sin() * 0.7) + ((i as f32 * 0.031).cos() * 0.2))
            .collect()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn runtime_v3() -> Option<pulp::x86::V3> {
        match Arch::new() {
            Arch::V3(simd) => Some(simd),
            _ => None,
        }
    }

    fn test_signal_f64(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as f64 * 0.13).sin() * 0.7) + ((i as f64 * 0.031).cos() * 0.2))
            .collect()
    }

    /// Parity for the f64 kernels production actually calls (the f32 set is
    /// retained as reference only). Lengths chosen to exercise vector tails,
    /// sub-vector buffers, and empty input.
    #[test]
    fn f64_kernels_dispatch_matches_scalar() {
        for len in [0usize, 1, 3, 7, 64, 129, 4_111, 65_537] {
            let data = test_signal_f64(len);

            assert!(
                (rms_f64(&data) - rms_f64_scalar(&data)).abs() <= REDUCTION_EPSILON,
                "rms_f64 parity at len {len}"
            );
            assert!(
                (peak_abs_f64(&data) - peak_abs_f64_scalar(&data)).abs() <= REDUCTION_EPSILON,
                "peak_abs_f64 parity at len {len}"
            );
            assert!(
                (sum_f64(&data) - sum_f64_dispatch_scalar(&data)).abs() <= REDUCTION_EPSILON,
                "sum_f64 parity at len {len}"
            );

            let mut a = data.clone();
            let mut b = data.clone();
            scale_in_place_f64(&mut a, 0.731);
            scale_in_place_f64_scalar(&mut b, 0.731);
            assert_eq!(a, b, "scale_in_place_f64 parity at len {len}");

            let mut a = data.clone();
            let mut b = data.clone();
            subtract_in_place_f64(&mut a, 0.0173);
            subtract_in_place_f64_scalar(&mut b, 0.0173);
            assert_eq!(a, b, "subtract_in_place_f64 parity at len {len}");

            let other = test_signal_f64(len);
            let mut a = data.clone();
            let mut b = data.clone();
            add_in_place_f64(&mut a, &other);
            add_in_place_f64_scalar(&mut b, &other);
            assert_eq!(a, b, "add_in_place_f64 parity at len {len}");
        }
    }

    #[test]
    fn f64_stereo_kernels_dispatch_matches_scalar() {
        for frames in [0usize, 1, 3, 64, 129, 2_049] {
            let interleaved = test_signal_f64(frames * 2);
            let (l_d, r_d) = deinterleave_stereo_f64(&interleaved);
            let (l_s, r_s) = deinterleave_stereo_f64_scalar(&interleaved);
            assert_eq!(l_d, l_s, "deinterleave L parity at {frames} frames");
            assert_eq!(r_d, r_s, "deinterleave R parity at {frames} frames");

            assert_eq!(
                interleave_stereo_f64(&l_d, &r_d),
                interleave_stereo_f64_scalar(&l_s, &r_s),
                "interleave parity at {frames} frames"
            );
            // Round trip is exact.
            assert_eq!(interleave_stereo_f64(&l_d, &r_d), interleaved);

            // Odd total length: both paths drop the dangling half-frame.
            let mut odd = interleaved.clone();
            odd.push(0.25);
            let (l_o, r_o) = deinterleave_stereo_f64(&odd);
            let (l_os, r_os) = deinterleave_stereo_f64_scalar(&odd);
            assert_eq!(l_o, l_os);
            assert_eq!(r_o, r_os);
            assert_eq!(l_o.len(), frames);
        }
    }

    /// Regression: out-of-range samples used to WRAP in the 24-bit packers
    /// (+1.5 encoded as bytes that decode to -0.5 — full polarity inversion).
    /// They must clamp, and rounding must match the FLAC writer's `.round()`.
    #[test]
    fn i24_quantizers_clamp_and_round() {
        let hot = [0.0f64, 0.5, 1.0, 1.5, -1.0, -1.5, 2.0e9, -2.0e9, f64::NAN];
        let bytes = scale_to_i24le_bytes_f64(&hot);
        let decoded = scale_from_i24le_bytes_f64(&bytes);
        let max_24 = 8_388_607.0 / 8_388_608.0; // +1.0 clamps to MAX_24/2^23
        let expect = [0.0f64, 0.5, max_24, max_24, -1.0, -1.0, max_24, -1.0, 0.0];
        for (i, (&got, &want)) in decoded.iter().zip(expect.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "i24 sample {i}: got {got}, want {want} (input {})",
                hot[i]
            );
        }
        // f32 variants share the same quantizer.
        let hot32: Vec<f32> = hot.iter().map(|&x| x as f32).collect();
        assert_eq!(scale_to_i24le_bytes_scalar(&hot32), bytes);
        assert_eq!(scale_to_i24le_bytes(&hot32), bytes);
    }

    #[test]
    fn mix_to_mono_dispatch_matches_scalar_exactly() {
        let mono = test_signal(129);
        assert_eq!(mix_to_mono(&mono, 1), mono);

        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s * 0.5]).collect();
        assert_eq!(mix_to_mono(&stereo, 2), mix_to_mono_scalar(&stereo, 2));
    }

    #[test]
    fn pearson_dispatch_matches_scalar_within_epsilon() {
        let a: Vec<f64> = (0..4097)
            .map(|i| ((i as f64 * 0.173).sin() * 0.8) + ((i as f64 * 0.017).cos() * 0.05))
            .collect();
        let b: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, &value)| value + ((i as f64 * 0.071).sin() * 0.0003))
            .collect();

        let scalar = pearson_correlation_scalar(&a, &b);
        let dispatched = pearson_correlation(&a, &b);
        assert!(
            (scalar - dispatched).abs() <= REDUCTION_EPSILON,
            "scalar={scalar}, dispatched={dispatched}"
        );
    }

    #[test]
    fn rms_dispatch_matches_scalar_within_epsilon() {
        let data = test_signal(65_537);
        let scalar = rms_f32_scalar(&data);
        let dispatched = rms_f32(&data);
        assert!(
            (scalar - dispatched).abs() <= F32_EPSILON,
            "scalar={scalar}, dispatched={dispatched}"
        );
    }

    #[test]
    fn scale_kernels_match_scalar_exactly() {
        let original = test_signal(4_111);

        let mut scaled_scalar = original.clone();
        scale_in_place_scalar(&mut scaled_scalar, 0.75);
        let mut scaled_dispatch = original.clone();
        scale_in_place(&mut scaled_dispatch, 0.75);
        assert_eq!(scaled_dispatch, scaled_scalar);

        let mut clamped_scalar = original.clone();
        scale_and_clamp_in_place_scalar(&mut clamped_scalar, 1.3, -0.8, 0.8);
        let mut clamped_dispatch = original;
        scale_and_clamp_in_place(&mut clamped_dispatch, 1.3, -0.8, 0.8);
        assert_eq!(clamped_dispatch, clamped_scalar);
    }

    #[test]
    fn pcm_scaling_matches_scalar_exactly() {
        let samples = test_signal(2049);
        assert_eq!(scale_to_i16(&samples), scale_to_i16_scalar(&samples));
        assert_eq!(
            scale_to_i24le_bytes(&samples),
            scale_to_i24le_bytes_scalar(&samples)
        );
        assert_eq!(
            scale_to_i32le_bytes(&samples),
            scale_to_i32le_bytes_scalar(&samples)
        );
    }

    #[test]
    fn peak_abs_matches_scalar() {
        let data = test_signal(12_345);
        assert_eq!(peak_abs_f32(&data), peak_abs_f32_scalar(&data));
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn direct_v3_path_matches_scalar_when_available() {
        let Some(simd) = runtime_v3() else {
            return;
        };

        let mono = test_signal(8193);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s * 0.75]).collect();
        let mix_scalar = mix_to_mono_scalar(&stereo, 2);
        let mix_simd = StereoMixToMono { data: &stereo }.with_simd(simd);
        assert_eq!(mix_simd, mix_scalar);

        let a: Vec<f64> = (0..5001).map(|i| (i as f64 * 0.017).sin()).collect();
        let b: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, &value)| value + ((i as f64 * 0.011).cos() * 0.0002))
            .collect();
        let pearson_scalar = pearson_correlation_scalar(&a, &b);
        let pearson_simd = PearsonCorrelation { a: &a, b: &b }.with_simd(simd);
        assert!(
            (pearson_scalar - pearson_simd).abs() <= REDUCTION_EPSILON,
            "scalar={pearson_scalar}, simd={pearson_simd}"
        );

        let rms_scalar = rms_f32_scalar(&mono);
        let rms_simd = RmsF32 { data: &mono }.with_simd(simd);
        assert!(
            (rms_scalar - rms_simd).abs() <= F32_EPSILON,
            "scalar={rms_scalar}, simd={rms_simd}"
        );

        // sum_f32
        let sum_scalar = sum_f32_scalar(&mono);
        let sum_simd = SumF32 { data: &mono }.with_simd(simd);
        assert!(
            (sum_scalar - sum_simd).abs() <= REDUCTION_EPSILON,
            "sum scalar={sum_scalar}, simd={sum_simd}"
        );

        // subtract_in_place
        let mut sub_scalar = test_signal(4_111);
        subtract_in_place_scalar(&mut sub_scalar, 0.42);
        let mut sub_simd = test_signal(4_111);
        SubtractInPlace {
            data: &mut sub_simd,
            value: 0.42,
        }
        .with_simd(simd);
        assert_eq!(sub_simd, sub_scalar);

        // deinterleave_stereo
        let (l_scalar, r_scalar) = deinterleave_stereo_scalar(&stereo);
        let (l_simd, r_simd) = DeinterleaveStereo { data: &stereo }.with_simd(simd);
        assert_eq!(l_simd, l_scalar);
        assert_eq!(r_simd, r_scalar);

        // interleave_stereo
        let interleaved_scalar = interleave_stereo_scalar(&l_scalar, &r_scalar);
        let interleaved_simd = InterleaveStereo {
            left: &l_scalar,
            right: &r_scalar,
        }
        .with_simd(simd);
        assert_eq!(interleaved_simd, interleaved_scalar);
    }

    #[test]
    fn sum_f32_dispatch_matches_scalar_within_epsilon() {
        let data = test_signal(65_537);
        let scalar = sum_f32_scalar(&data);
        let dispatched = sum_f32(&data);
        assert!(
            (scalar - dispatched).abs() <= REDUCTION_EPSILON,
            "scalar={scalar}, dispatched={dispatched}"
        );
    }

    #[test]
    fn subtract_in_place_dispatch_matches_scalar_exactly() {
        let mut scalar_buf = test_signal(4_111);
        subtract_in_place_scalar(&mut scalar_buf, 0.42);
        let mut dispatch_buf = test_signal(4_111);
        subtract_in_place(&mut dispatch_buf, 0.42);
        assert_eq!(dispatch_buf, scalar_buf);
    }

    #[test]
    fn scale_from_i24le_bytes_roundtrip() {
        let original = test_signal(8_193);
        let encoded = scale_to_i24le_bytes_scalar(&original);
        let decoded = scale_from_i24le_bytes(&encoded);
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() <= 2e-7, "orig={orig}, decoded={dec}");
        }
    }

    #[test]
    fn scale_from_i24le_bytes_dispatch_matches_scalar() {
        let original = test_signal(4_097);
        let encoded = scale_to_i24le_bytes_scalar(&original);
        let scalar = scale_from_i24le_bytes_scalar(&encoded);
        let dispatched = scale_from_i24le_bytes(&encoded);
        assert_eq!(dispatched, scalar);
    }

    #[test]
    fn deinterleave_interleave_roundtrip_exact() {
        let stereo = test_signal(8_192);
        let (left, right) = deinterleave_stereo(&stereo);
        let roundtrip = interleave_stereo(&left, &right);
        assert_eq!(roundtrip, stereo);
    }

    #[test]
    fn deinterleave_dispatch_matches_scalar_exactly() {
        let stereo = test_signal(8_193);
        let (l_scalar, r_scalar) = deinterleave_stereo_scalar(&stereo);
        let (l_dispatch, r_dispatch) = deinterleave_stereo(&stereo);
        assert_eq!(l_dispatch, l_scalar);
        assert_eq!(r_dispatch, r_scalar);

        assert_eq!(
            interleave_stereo(&l_scalar, &r_scalar),
            interleave_stereo_scalar(&l_scalar, &r_scalar)
        );
    }
}
