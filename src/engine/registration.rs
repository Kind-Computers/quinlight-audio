// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

//! Registration-based consensus — 1D port of the dense Lucas-Kanade
//! "register consensus" from the `up` image project
//! (`up/crates/up-consensus/src/registration_cpu.rs`).
//!
//! Pipeline shape, per channel (mono f64 @ 48 kHz):
//!
//! 1. The **master** is the sinc-resampled 48 kHz reference — the most
//!    mathematically accurate upsample (libswresample polyphase Kaiser ≈
//!    Whittaker-Shannon). It defines the common time grid and carries the
//!    exact loop length / loop point.
//! 2. For every AI engine, compute a dense 1D optical-flow field mapping the
//!    master's samples to that engine's, then linearly warp the engine onto
//!    the master's grid (`dense_lk_1d` + `linear_warp_1d`).
//! 3. Per-sample **median** (default) or **mean** across the warped engine
//!    stack.
//!
//! **The master is the warp/flow target ONLY — it is never a member of the
//! reduction stack.** It is band-limited to the source Nyquist (no real
//! high-frequency content), so including it would dilute the AI-generated
//! highs. The reduction is over the warped engines alone.
//!
//! Why this replaces the frequency-domain `spectral_intersection`: that path
//! manipulates per-bin STFT magnitude/phase and inverse-transforms, which
//! rings in the time domain (pre-echo, transient smearing). Warp-then-reduce
//! is purely time-domain — no FFT, no per-bin phase averaging — so it does
//! not ring, and because every engine is warped onto the master's exact loop
//! timing the loop seam stays sample-accurate.
//!
//! The 2D structure-tensor solve reduces exactly to 1D: with no y axis the
//! `syy`, `sxy`, `syt` terms vanish and the 2×2 inverse collapses to the
//! scalar `u = −Σ(w·Iₓ·Iₜ) / Σ(w·Iₓ²)`.

/// Per-sample reduction across the warped engine stack.
///
/// `Median` (default) is outlier-robust: a single-engine hallucination at a
/// sample is rejected when the other K−1 engines agree. `Mean` only
/// attenuates an outlier by `1/K` rather than rejecting it — smoother on
/// content the engines agree on, at the cost of outlier sensitivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReductionMode {
    #[default]
    Median,
    Mean,
}

/// Tunables for [`dense_lk_1d`]. [`Default`] reproduces the recommended
/// single-level configuration.
#[derive(Debug, Clone, Copy)]
pub struct LkParams {
    /// Window radius in samples (window is `2*radius + 1` taps). Wider than
    /// the 2D port's radius 3 because the master is band-limited to the
    /// source Nyquist, so the window must span ≳ one cycle of the highest
    /// master component for a well-conditioned `Σ w·Iₓ²`.
    pub radius: usize,
    /// Forward-additive warp-refinement passes. `1` = a single linearised
    /// solve (exact parity with the 2D single-level path). `>1` re-warps the
    /// original engine by the cumulative flow and re-solves each pass — buys
    /// robustness against 2–4 sample drift at linear cost. No pyramid.
    pub n_iters: usize,
    /// Structure floor on `Σ w·Iₓ²`, relative to local master energy
    /// (`Σ w·m²`). Makes the "enough gradient to trust the flow" bar
    /// level-invariant (a quiet passage registers like a loud one).
    pub det_floor_rel: f64,
    /// Absolute structure floor — a numerical guard against a `1/sxx` blow-up
    /// at near-silence. Must sit far below `det_floor_rel * e_local` across
    /// the usable range.
    pub det_floor_abs: f64,
    /// Goodness-of-fit gate. The flow is trusted at a sample only when the LK
    /// model `Iₜ ≈ −u·Iₓ` explains at least this fraction of the local
    /// difference energy, i.e. `R² = sxt² / (sxx · Σ w·Iₜ²) ≥ min_r2`. This is
    /// the weighted correlation between the master gradient and the
    /// master-vs-engine difference. It is the crucial robustness term for a
    /// **band-limited** master: when `Iₜ` is dominated by high-frequency
    /// content the master cannot represent (the AI-generated band above the
    /// source Nyquist), `R²` is low and the flow is rejected (`u = 0`,
    /// identity warp) rather than letting that HF drive a spurious shift that
    /// would mangle the engine's own HF. When the engines genuinely differ by
    /// an in-band time shift, `R² ≈ 1` and the flow is trusted.
    pub min_r2: f64,
    /// Use a fixed **spatial** Gaussian window taper (`exp(−d²/2σ²)`,
    /// `σ = radius/2`) instead of a uniform window. Value-independent, so
    /// none of the 2D bilateral weight's bipolar-audio problems. Off by
    /// default.
    pub gaussian_window: bool,
}

impl Default for LkParams {
    fn default() -> Self {
        Self {
            radius: 5,
            n_iters: 1,
            det_floor_rel: 1e-3,
            det_floor_abs: 1e-12,
            min_r2: 0.5,
            gaussian_window: false,
        }
    }
}

/// Fold a non-finite (NaN / ±Inf) sample to 0 before it enters the reduction
/// so a single bad engine value can't poison the median (NaN sorts
/// unpredictably) or NaN the mean. Audio is bipolar, so — unlike the 2D
/// image port's `scrub1` — negatives pass through unchanged (no
/// rectification) and there is no photometric ceiling clamp; the downstream
/// RMS normalisation handles level.
#[inline]
fn scrub1(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Loop-aware index into a length-`n` buffer. `looped` wraps circularly
/// (`rem_euclid`) so a window or warp straddling the loop seam reads across
/// it continuously — the per-sample analogue of `spectral::pad_for_stft`'s
/// looped wrap. Non-looped clamps to the edge (replicate), like the 2D port.
#[inline]
fn idx(j: isize, n: usize, looped: bool) -> usize {
    debug_assert!(n >= 1);
    if looped {
        j.rem_euclid(n as isize) as usize
    } else {
        j.clamp(0, n as isize - 1) as usize
    }
}

/// Window tap weight. Uniform by default; an optional fixed spatial Gaussian
/// taper reduces flow-field blockiness as the window slides. Never depends on
/// sample value (the 2D bilateral `log2(luma)` weight does not port to
/// bipolar audio).
#[inline]
fn window_weight(d: isize, radius: isize, gaussian: bool) -> f64 {
    if gaussian {
        let sigma = (radius as f64) * 0.5;
        let x = d as f64;
        (-(x * x) / (2.0 * sigma * sigma)).exp()
    } else {
        1.0
    }
}

/// Sample `signal` at sub-sample position `pos` with linear interpolation.
/// `looped` wraps at the buffer ends (loop continuity); otherwise clamps to
/// the edge. A non-finite `pos` returns 0 (defensive; the warp also guards
/// the flow vector).
#[inline]
fn sample_linear(signal: &[f64], pos: f64, looped: bool) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return signal[0];
    }
    if !pos.is_finite() {
        return 0.0;
    }
    let base = pos.floor();
    let frac = pos - base;
    let i0 = base as isize;
    let (a, b) = if looped {
        (
            signal[i0.rem_euclid(n as isize) as usize],
            signal[(i0 + 1).rem_euclid(n as isize) as usize],
        )
    } else {
        (
            signal[i0.clamp(0, n as isize - 1) as usize],
            signal[(i0 + 1).clamp(0, n as isize - 1) as usize],
        )
    };
    a + (b - a) * frac
}

/// Single-level (or `n_iters`-refined) dense 1D Lucas-Kanade flow FROM the
/// `master` TO the `engine`.
///
/// Per output sample `i`, over a window of radius `params.radius`:
/// `Iₓ` is the central difference of the **master**, `Iₜ = engine − master`,
/// and the flow is `u = −Σ(w·Iₓ·Iₜ) / Σ(w·Iₓ²)`, floored against
/// `max(det_floor_abs, det_floor_rel · Σ w·master²)` (zero flow below the
/// floor — flat / silent regions). The sign is such that
/// `sample_linear(engine, i + u[i])` pulls the engine feature back onto the
/// master's grid: an engine running `s` samples late yields `u ≈ +s`.
///
/// Returns one `u` per sample, length `min(master.len(), engine.len())`.
pub fn dense_lk_1d(master: &[f64], engine: &[f64], looped: bool, params: LkParams) -> Vec<f64> {
    let n = master.len().min(engine.len());
    let mut flow = vec![0.0f64; n];
    // Central differences need at least 3 samples; below that the flow is
    // undefined — leave it at identity.
    if n < 3 {
        return flow;
    }
    let r = params.radius.max(1) as isize;
    let iters = params.n_iters.max(1);

    // `warped` holds the engine re-warped by the cumulative flow between
    // refinement passes; pass 0 sees the raw engine.
    let mut warped: Vec<f64> = engine[..n].to_vec();
    for iter in 0..iters {
        for (i, flow_i) in flow.iter_mut().enumerate() {
            let ii = i as isize;
            let mut sxx = 0.0f64;
            let mut sxt = 0.0f64;
            let mut stt = 0.0f64;
            let mut e_local = 0.0f64;
            for d in -r..=r {
                let j = ii + d;
                let mc = master[idx(j, n, looped)];
                let ix = 0.5 * (master[idx(j + 1, n, looped)] - master[idx(j - 1, n, looped)]);
                let it = warped[idx(j, n, looped)] - mc;
                let w = window_weight(d, r, params.gaussian_window);
                sxx += w * ix * ix;
                sxt += w * ix * it;
                stt += w * it * it;
                e_local += w * mc * mc;
            }
            let det_floor = params.det_floor_abs.max(params.det_floor_rel * e_local);
            // Goodness-of-fit gate: trust the flow only when the model
            // `It ≈ -u·Ix` explains ≥ min_r2 of the local difference energy,
            // i.e. R² = sxt²/(sxx·stt) ≥ min_r2. Rearranged to avoid division
            // and to stay well-defined when stt → 0 (then RHS → 0, LHS ≥ 0, so
            // the tiny residual flow — sxt ≈ 0 — is allowed through as ~0).
            let fit_ok = sxt * sxt >= params.min_r2 * sxx * stt;
            let du = if sxx > det_floor && fit_ok {
                -sxt / sxx
            } else {
                0.0
            };
            *flow_i += if du.is_finite() { du } else { 0.0 };
        }
        // Re-warp the ORIGINAL engine by the cumulative flow for the next
        // pass (forward-additive). Skip on the final pass.
        if iter + 1 < iters {
            warped = linear_warp_1d(&engine[..n], &flow, looped);
        }
    }
    flow
}

/// Warp `signal` by a per-sample flow field: `out[i] = signal[i + flow[i]]`
/// with linear interpolation. `looped` selects circular wrap vs edge clamp.
/// A non-finite `flow[i]` is treated as 0 (identity at that sample).
/// Output length is `min(signal.len(), flow.len())`.
pub fn linear_warp_1d(signal: &[f64], flow: &[f64], looped: bool) -> Vec<f64> {
    let n = signal.len().min(flow.len());
    let mut out = vec![0.0f64; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let u = if flow[i].is_finite() { flow[i] } else { 0.0 };
        *slot = sample_linear(signal, i as f64 + u, looped);
    }
    out
}

/// Per-sample median across the warped engine stack (master excluded).
/// Each value is scrubbed (NaN/Inf → 0) on gather. Even `K` averages the two
/// middle survivors; odd `K` takes the middle. `K == 1` passes the lone
/// engine through; `K == 0` returns empty.
pub fn median_per_sample(stack: &[&[f64]]) -> Vec<f64> {
    let k = stack.len();
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return stack[0].to_vec();
    }
    let n = stack.iter().map(|s| s.len()).min().unwrap_or(0);
    let mut out = vec![0.0f64; n];
    let mut buf: Vec<f64> = Vec::with_capacity(k);
    let mid = k / 2;
    for (i, slot) in out.iter_mut().enumerate() {
        buf.clear();
        for s in stack {
            buf.push(scrub1(s[i]));
        }
        // NaN already scrubbed to 0, but keep a total order for safety.
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        *slot = if k.is_multiple_of(2) {
            0.5 * (buf[mid - 1] + buf[mid])
        } else {
            buf[mid]
        };
    }
    out
}

/// Per-sample arithmetic mean across the warped engine stack (master
/// excluded). Scrubs each value on gather so a single NaN/Inf can't poison
/// the average. `K == 1` passes through; `K == 0` returns empty.
pub fn mean_per_sample(stack: &[&[f64]]) -> Vec<f64> {
    let k = stack.len();
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return stack[0].to_vec();
    }
    let n = stack.iter().map(|s| s.len()).min().unwrap_or(0);
    let inv = 1.0 / k as f64;
    let mut out = vec![0.0f64; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for s in stack {
            acc += scrub1(s[i]);
        }
        *slot = acc * inv;
    }
    out
}

/// End-to-end 1D registration consensus with [`LkParams::default`].
///
/// For each engine: compute LK flow master→engine, linearly warp the engine
/// onto the master grid, and push the warped engine. **The master is not
/// pushed** — it is the warp target only. The K-engine stack is reduced
/// per-sample by `mode`. `K == 1` returns the lone engine unchanged (no warp,
/// no master mixing); `K == 0` returns empty. Output length is the minimum
/// engine length.
pub fn registration_consensus_1d(
    master: &[f64],
    engines: &[&[f64]],
    looped: bool,
    mode: ReductionMode,
) -> Vec<f64> {
    registration_consensus_1d_with(master, engines, looped, mode, LkParams::default())
}

/// [`registration_consensus_1d`] with explicit [`LkParams`].
pub fn registration_consensus_1d_with(
    master: &[f64],
    engines: &[&[f64]],
    looped: bool,
    mode: ReductionMode,
    params: LkParams,
) -> Vec<f64> {
    let k = engines.len();
    if k == 0 {
        return Vec::new();
    }
    let out_len = engines.iter().map(|e| e.len()).min().unwrap_or(0);
    if k == 1 {
        return engines[0][..out_len].to_vec();
    }

    // Alignment is bounded by the master too; if the master is shorter than
    // the engines (or empty/too short) the unaligned tail falls through with
    // identity flow rather than being dropped.
    let align_n = master.len().min(out_len);

    let warped: Vec<Vec<f64>> = engines
        .iter()
        .map(|e| {
            let en = &e[..out_len];
            if align_n >= 3 {
                let flow_short = dense_lk_1d(master, en, looped, params);
                let mut flow = vec![0.0f64; out_len];
                flow[..align_n].copy_from_slice(&flow_short[..align_n]);
                linear_warp_1d(en, &flow, looped)
            } else {
                en.to_vec()
            }
        })
        .collect();

    let refs: Vec<&[f64]> = warped.iter().map(|v| v.as_slice()).collect();
    match mode {
        ReductionMode::Median => median_per_sample(&refs),
        ReductionMode::Mean => mean_per_sample(&refs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sr: f64, n: usize, freq: f64, amp: f64, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr + phase).sin())
            .collect()
    }

    fn rms(s: &[f64]) -> f64 {
        if s.is_empty() {
            return 0.0;
        }
        (s.iter().map(|x| x * x).sum::<f64>() / s.len() as f64).sqrt()
    }

    fn mean_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        assert!(n > 0);
        a.iter()
            .zip(b.iter())
            .take(n)
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>()
            / n as f64
    }

    /// Amplitude of a single frequency component (single-bin Goertzel).
    fn component_amplitude(signal: &[f64], freq: f64, sr: f64) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        let mut sin_dot = 0.0;
        let mut cos_dot = 0.0;
        for (i, &s) in signal.iter().enumerate() {
            let ph = 2.0 * std::f64::consts::PI * freq * i as f64 / sr;
            sin_dot += s * ph.sin();
            cos_dot += s * ph.cos();
        }
        2.0 * (sin_dot * sin_dot + cos_dot * cos_dot).sqrt() / signal.len() as f64
    }

    /// Correlation of a signal against a unit reference (gain-normalised),
    /// used to measure how much of a given frequency leaked into the output.
    fn correlation(signal: &[f64], reference: &[f64]) -> f64 {
        let n = signal.len().min(reference.len());
        let dot: f64 = signal.iter().zip(reference).map(|(a, b)| a * b).sum();
        let ns: f64 = signal[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
        let nr: f64 = reference[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
        if ns > 0.0 && nr > 0.0 {
            dot / (ns * nr)
        } else {
            0.0
        }
    }

    /// Sub-sample shift of `sig` to the right by `s` samples via linear
    /// interpolation: `out[i] = sig(i − s)`.
    fn shift(sig: &[f64], s: f64, looped: bool) -> Vec<f64> {
        let flow = vec![-s; sig.len()];
        linear_warp_1d(sig, &flow, looped)
    }

    const SR: f64 = 48_000.0;

    #[test]
    fn identity_engines_passthrough() {
        // Three identical engines, master == engine signal → zero flow,
        // output == engine exactly. Round-trip stability.
        let n = 2048;
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                0.6 * (2.0 * std::f64::consts::PI * 1500.0 * i as f64 / SR).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 4000.0 * i as f64 / SR).sin()
            })
            .collect();
        let refs = [sig.as_slice(), sig.as_slice(), sig.as_slice()];
        let out = registration_consensus_1d(&sig, &refs, false, ReductionMode::Median);
        assert_eq!(out.len(), n);
        assert!(
            mean_abs_diff(&out, &sig) < 1e-9,
            "identical engines with matching master should pass through, diff={}",
            mean_abs_diff(&out, &sig)
        );
    }

    #[test]
    fn n1_returns_engine() {
        let n = 512;
        let engine = tone(SR, n, 440.0, 0.5, 0.0);
        let master = tone(SR, n, 880.0, 0.9, 0.0);
        let out = registration_consensus_1d(&master, &[engine.as_slice()], false, ReductionMode::Median);
        assert_eq!(out, engine, "single engine must pass through unchanged");
    }

    #[test]
    fn median_rejects_single_engine_outlier() {
        // 4 engines agree on a 440 Hz tone; 1 adds a 3 kHz component. Master
        // is the clean 440 Hz tone. The lone 3 kHz outlier is rejected by the
        // median, so it should be ~absent from the output.
        let n = 8192;
        let clean = tone(SR, n, 440.0, 0.5, 0.0);
        let three_k = tone(SR, n, 3000.0, 0.5, 0.0);
        let dirty: Vec<f64> = clean.iter().zip(&three_k).map(|(a, b)| a + b).collect();
        let refs = [
            clean.as_slice(),
            clean.as_slice(),
            clean.as_slice(),
            clean.as_slice(),
            dirty.as_slice(),
        ];
        let out = registration_consensus_1d(&clean, &refs, false, ReductionMode::Median);
        let leak = correlation(&out, &three_k).abs();
        assert!(leak < 0.15, "3 kHz outlier should be rejected, leak={leak}");
    }

    #[test]
    fn subsample_shift_no_ringing_no_overshoot() {
        // Headline anti-ringing test. Engines are fractional-sample shifts
        // {0, +0.3, +0.7} of a band-limited two-tone signal; master is the
        // unshifted version. Registration warps each engine back onto the
        // master grid, so the median reconstructs the signal with NO
        // overshoot past the input peak (the Gibbs ringing the STFT path
        // exhibits).
        let n = 1024;
        let base: Vec<f64> = (0..n)
            .map(|i| {
                0.6 * (2.0 * std::f64::consts::PI * 1500.0 * i as f64 / SR).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 3000.0 * i as f64 / SR).sin()
            })
            .collect();
        let e0 = base.clone();
        let e1 = shift(&base, 0.3, false);
        let e2 = shift(&base, 0.7, false);
        let refs = [e0.as_slice(), e1.as_slice(), e2.as_slice()];
        let out = registration_consensus_1d(&base, &refs, false, ReductionMode::Median);

        let in_peak = base.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        // Check the interior (away from the edge-clamp transients).
        let margin = 32;
        let out_peak = out[margin..n - margin]
            .iter()
            .fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(
            out_peak <= in_peak * 1.02,
            "registration overshoot: out_peak={out_peak} vs in_peak={in_peak}"
        );
        // And the aligned result tracks the master closely (engines warped
        // back into register), much better than a naive average would.
        let interior_diff = mean_abs_diff(&out[margin..n - margin], &base[margin..n - margin]);
        assert!(
            interior_diff < 0.05 * rms(&base),
            "aligned median should track the unshifted signal, diff={interior_diff}"
        );
    }

    #[test]
    fn loop_wrap_seam_continuity() {
        // Integer-cycle tone so the loop seam is continuous. With looped=true
        // the warp/window wrap across the seam, so the output stays
        // continuous end-to-start.
        let n = 4096;
        let cycles = (440.0 * n as f64 / SR).round();
        let freq = cycles * SR / n as f64;
        let clean = tone(SR, n, freq, 0.5, 0.0);
        let e1: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.01 * (i as f64 * 3.7).sin())
            .collect();
        let e2: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.01 * (i as f64 * 5.1).cos())
            .collect();
        let out = registration_consensus_1d(&clean, &[e1.as_slice(), e2.as_slice()], true, ReductionMode::Median);
        let peak = out.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        let gap = (out[out.len() - 1] - out[0]).abs();
        let rel = if peak > 0.0 { gap / peak } else { 0.0 };
        assert!(rel < 0.1, "loop seam should stay continuous, rel_gap={rel}");
    }

    #[test]
    fn master_excluded_from_reduction() {
        // The master is a band-limited (440 Hz only) reference — the audio
        // analogue of the sinc upsample, which has no real HF. The engines
        // all carry an extra 6 kHz component. Because the master is the warp
        // target ONLY (never in the reduction stack) and the engines share
        // the master's 440 Hz (so the flow is ~0), the engines' 6 kHz
        // survives at full amplitude. Reduction is Mean: were the master
        // wrongly included in the stack, the mean would dilute the 6 kHz
        // toward the master's HF-free value (here by 1/3), which this asserts
        // against.
        let n = 8192;
        let s: Vec<f64> = (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin()
                    + 0.4 * (2.0 * std::f64::consts::PI * 6000.0 * i as f64 / SR).sin()
            })
            .collect();
        let master = tone(SR, n, 440.0, 0.5, 0.0); // band-limited: no 6 kHz
        let refs = [s.as_slice(), s.as_slice()];
        let out = registration_consensus_1d(&master, &refs, false, ReductionMode::Mean);

        let amp_engine = component_amplitude(&s, 6000.0, SR);
        let amp_out = component_amplitude(&out, 6000.0, SR);
        assert!(
            amp_out > 0.9 * amp_engine,
            "engines' 6 kHz must survive (master is excluded from the stack): \
             out={amp_out:.4}, engine={amp_engine:.4}"
        );
        // And the master's own band (440 Hz) is still present (shared content),
        // i.e. the output tracks the engines, not a hollowed-out signal.
        assert!(
            mean_abs_diff(&out, &s) < 0.05 * rms(&s),
            "output should track the engines (warp ~ identity on shared content)"
        );
    }

    #[test]
    fn mean_attenuates_outlier_by_1_over_n() {
        // Constant master/engines → zero gradient → zero flow → identity
        // warp. A single-sample spike in one of 5 engines is attenuated by
        // 1/5 under Mean (not rejected), the documented mean-vs-median
        // asymmetry.
        let n = 64;
        let base = vec![0.3f64; n];
        let mut spiky = base.clone();
        let p = 17;
        spiky[p] = 5.0;
        let refs = [
            base.as_slice(),
            base.as_slice(),
            base.as_slice(),
            base.as_slice(),
            spiky.as_slice(),
        ];
        let out = registration_consensus_1d(&base, &refs, false, ReductionMode::Mean);
        let expected = (4.0 * 0.3 + 5.0) / 5.0;
        assert!(
            (out[p] - expected).abs() < 1e-9,
            "mean should attenuate the spike by 1/N: expected {expected}, got {}",
            out[p]
        );
        assert!((out[0] - 0.3).abs() < 1e-9, "untouched sample stays at base");
    }

    #[test]
    fn master_unavailable_or_short() {
        // Empty and too-short masters must not panic; output length is the
        // min engine length and the engines reduce un-warped.
        let n = 256;
        let e1 = tone(SR, n, 440.0, 0.5, 0.0);
        let e2 = tone(SR, n, 440.0, 0.5, 0.1);

        let empty: Vec<f64> = Vec::new();
        let out = registration_consensus_1d(&empty, &[e1.as_slice(), e2.as_slice()], false, ReductionMode::Median);
        assert_eq!(out.len(), n);

        let short = vec![0.1, 0.2];
        let out2 = registration_consensus_1d(&short, &[e1.as_slice(), e2.as_slice()], true, ReductionMode::Median);
        assert_eq!(out2.len(), n);
    }

    #[test]
    fn dense_lk_recovers_subsample_shift() {
        // Direct flow check: engine shifted right by +0.4 samples vs master →
        // recovered flow u ≈ +0.4 in the interior.
        let n = 1024;
        let master: Vec<f64> = (0..n)
            .map(|i| {
                0.6 * (2.0 * std::f64::consts::PI * 1500.0 * i as f64 / SR).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 3500.0 * i as f64 / SR).sin()
            })
            .collect();
        let engine = shift(&master, 0.4, false);
        let flow = dense_lk_1d(&master, &engine, false, LkParams::default());
        let margin = 16;
        let mean_u = flow[margin..n - margin].iter().sum::<f64>() / (n - 2 * margin) as f64;
        assert!(
            (mean_u - 0.4).abs() < 0.1,
            "dense_lk_1d should recover the +0.4 sample shift, got {mean_u}"
        );
    }
}
