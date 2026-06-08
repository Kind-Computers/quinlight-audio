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
//! 3. Per-sample **mean**, **median**, or **robust** reduction across the warped engine
//!    stack.
//!
//! The **master** is the warp/flow target (and is never itself warped). In
//! **median** reduction it is *also* a full member of the stack — a
//! self-regulating one: band-limited to the source Nyquist, it sits among the
//! engines in the source band (anchoring/guiding the consensus and helping
//! outvote an in-band hallucination) but is a low outlier in the AI-extended
//! high band, where the median rejects it, so the band extension is preserved.
//! In **mean** and **robust** modes it stays excluded, since blending in its
//! zero highs would dilute the AI-generated band extension.
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
/// `Median` is outlier-robust: a single-engine hallucination at a sample is
/// rejected when the other K−1 engines agree. `Mean` only attenuates an
/// outlier by `1/K` rather than rejecting it — smoother on content the engines
/// agree on, at the cost of outlier sensitivity. `Robust` is the winsorized
/// middle ground and the **shipped default** (selected by the process static;
/// see `set_quinlight_registration_mode`): it clamps each engine to the
/// per-sample median ± a MAD-scaled tolerance before averaging, keeping the
/// mean's smoothing where the engines agree yet rejecting a lone spike like the
/// median. See [`robust_mean_per_sample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReductionMode {
    #[default]
    Median,
    Mean,
    Robust,
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
    /// `σ = radius/2`) instead of a uniform window. Value-independent and
    /// composes (multiplicatively) with the amplitude bilateral weight
    /// ([`LkParams::sigma_a`]). Off by default.
    pub gaussian_window: bool,
    /// Bipolar **bilateral (edge) weight** bandwidth, in units of the
    /// window-local master amplitude scale (≈ the plateau half-swing of a
    /// square wave). A tap whose master value differs from the window centre by
    /// `sigma_a · scale` is attenuated to `e^{−1/2}`, so a window straddling a
    /// square-wave transition solves one plateau at a time instead of a smeared
    /// blend across both. Normalised by a window-local master variance
    /// ([`bilateral_scale2`]), so it is level-invariant — a loud and a quiet
    /// square weight identically. This is the bipolar analogue of the 2D image
    /// port's `log2(luma)` bilateral (which does not apply to signed audio).
    /// `f64::INFINITY` disables it (pure spatial weighting — the prior
    /// behaviour).
    pub sigma_a: f64,
    /// Numerical guard added to the bilateral-weight denominator so a flat /
    /// silent window (variance → 0) yields `w_photo ≈ 1` (uniform) rather than a
    /// `0/0`. Must sit far below `2·sigma_a²·var` across the usable range.
    pub bilateral_scale_eps: f64,
    /// Loop-**seam guard** half-width, in samples, around each loop endpoint
    /// (local indices `0` and `n−1`) when `looped`. `0` disables it. Within the
    /// guard the flow is pinned to identity (registration never *moves* samples
    /// at the seam) and the reduction is blended toward the band-limited master,
    /// which attenuates a seam tick **shared across all engines** — such ticks
    /// survive the per-sample median — and restores exact loop continuity. On
    /// by default (`8`); set to `0` to disable. Self-gates on `looped`, so it is
    /// inert for one-shot (non-looped) samples.
    pub seam_guard_radius: usize,
    /// Include the band-limited sinc **master** as a full member of the
    /// per-sample **median** reduction (in addition to being the warp target),
    /// giving the AI engines a mathematically-correct reference vote. It is
    /// self-regulating: in the source band it sits among the engines and guides
    /// the consensus (helping reject an in-band hallucination), while in the
    /// AI-extended high band it is ≈ 0 — a low outlier the median discards — so
    /// the band extension survives. **Median only**: under `Mean` and `Robust`
    /// the master stays excluded (Mean cannot reject it, and Robust keeps the
    /// AI-extended highs undiluted), regardless of this flag. On by default.
    pub include_master_in_reduction: bool,
    /// Winsor window slope (multiples of the per-sample MAD) for the
    /// [`ReductionMode::Robust`] reducer: `tol = winsor_k·MAD + winsor_abs`.
    /// `0` ⇒ a fixed ±`winsor_abs` band; large ⇒ the plain mean. Inert for
    /// `Median`/`Mean`.
    pub winsor_k: f64,
    /// Winsor absolute floor in full-scale units (audio ≈ [-1, 1]) for the
    /// [`ReductionMode::Robust`] reducer — keeps the clamp window non-zero when
    /// the engines agree exactly (MAD = 0) so the mean's `σ/√N` smoothing
    /// survives. Inert for `Median`/`Mean`.
    pub winsor_abs: f64,
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
            sigma_a: 0.6,
            bilateral_scale_eps: 1e-12,
            seam_guard_radius: 8,
            include_master_in_reduction: true,
            winsor_k: 2.5,
            winsor_abs: 0.01,
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

/// Window-local variance of the `master` about its window mean — the
/// normaliser for the bipolar bilateral edge weight in [`dense_lk_1d`]. Makes
/// the weight level-invariant: a rail-to-rail square gives `var ≈ A²`, a quiet
/// square `≈ a²`, so the *normalised* tap-to-centre differences (and hence the
/// weights) match. One extra pass over the same `2*radius+1` taps the flow
/// solve already visits.
#[inline]
fn bilateral_scale2(master: &[f64], i: usize, radius: isize, n: usize, looped: bool) -> f64 {
    let ii = i as isize;
    let count = (2 * radius + 1) as f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for d in -radius..=radius {
        let m = master[idx(ii + d, n, looped)];
        s1 += m;
        s2 += m * m;
    }
    let mean = s1 / count;
    (s2 / count - mean * mean).max(0.0)
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
            // Bipolar bilateral (edge) weight: taps whose master value differs
            // from the window centre are attenuated, so a window straddling a
            // square-wave transition solves one plateau at a time instead of a
            // smeared blend. Normalised by the window-local master variance, so
            // it is level-invariant. `sigma_a = INFINITY` ⇒ `inv_2s2 = 0` ⇒
            // `w_photo ≡ 1` (pure spatial weighting, the prior behaviour).
            let mc0 = master[idx(ii, n, looped)];
            let inv_2s2 = if params.sigma_a.is_finite() {
                let var = bilateral_scale2(master, i, r, n, looped);
                1.0 / (2.0 * params.sigma_a * params.sigma_a * var + params.bilateral_scale_eps)
            } else {
                0.0
            };
            let mut sxx = 0.0f64;
            let mut sxt = 0.0f64;
            let mut stt = 0.0f64;
            let mut e_local = 0.0f64;
            for d in -r..=r {
                let j = ii + d;
                let mc = master[idx(j, n, looped)];
                let ix = 0.5 * (master[idx(j + 1, n, looped)] - master[idx(j - 1, n, looped)]);
                let it = warped[idx(j, n, looped)] - mc;
                // Spatial × amplitude-bilateral weight. The centre tap (`d=0`,
                // `da=0`) always keeps `w_photo=1`, so the solve never loses the
                // sample it is estimating.
                let w_photo = if inv_2s2 > 0.0 {
                    let da = mc - mc0;
                    (-(da * da) * inv_2s2).exp()
                } else {
                    1.0
                };
                let w = window_weight(d, r, params.gaussian_window) * w_photo;
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
            let mut du = if sxx > det_floor && fit_ok {
                -sxt / sxx
            } else {
                0.0
            };
            // Loop-seam guard (flow pin): within `seam_guard_radius` of either
            // loop endpoint, hold the flow at identity. Registration must not
            // *move* samples at the seam — any motion there manufactures a seam
            // mismatch. Inert unless looped and the radius is set.
            if params.seam_guard_radius > 0 && looped {
                let g = params.seam_guard_radius;
                if i < g || i >= n.saturating_sub(g) {
                    du = 0.0;
                }
            }
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

/// Per-sample **winsorized (robust) mean** across the warped engine stack
/// (master excluded, exactly like [`mean_per_sample`]). At each sample: take
/// the per-sample `center` (median), the median absolute deviation
/// `mad = median(|xᵢ − center|)`, form the window `tol = winsor_k·mad +
/// winsor_abs`, clamp every value to `[center−tol, center+tol]`, and average. A
/// lone-engine spike that disagrees with the others by ≫ `mad` is bounded to
/// the window (≈ rejected like a median), while agreed samples sit inside the
/// window and pass through unchanged, so the result keeps the plain mean's
/// `σ/√N` smoothing. `winsor_k → 0` ⇒ a fixed ±`winsor_abs` band (and, with
/// `winsor_abs → 0`, mean ≈ median); `winsor_k → ∞` ⇒ the plain mean.
///
/// The window is **MAD-scaled** (sign-agnostic) rather than magnitude-relative
/// like the 2D image port's `WINSOR_TAU·median.max(0)` term: audio is bipolar,
/// so a magnitude window would be zero on every negative half-wave and would
/// key off loudness instead of cross-engine disagreement. Each value is
/// scrubbed (NaN/Inf → 0) on gather. `K == 1` passes through; `K == 0` returns
/// empty.
pub fn robust_mean_per_sample(stack: &[&[f64]], winsor_k: f64, winsor_abs: f64) -> Vec<f64> {
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
    let mut buf: Vec<f64> = Vec::with_capacity(k); // engine values (sorted for the median)
    let mut dev: Vec<f64> = Vec::with_capacity(k); // |value − center| (sorted for the MAD)
    let mid = k / 2;
    let even = k.is_multiple_of(2);
    for (i, slot) in out.iter_mut().enumerate() {
        buf.clear();
        for s in stack {
            buf.push(scrub1(s[i]));
        }
        // center = median (same total-order + even/odd tie-break as `median_per_sample`).
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        let center = if even {
            0.5 * (buf[mid - 1] + buf[mid])
        } else {
            buf[mid]
        };
        // mad = median absolute deviation about the center.
        dev.clear();
        for &v in buf.iter() {
            dev.push((v - center).abs());
        }
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        let mad = if even {
            0.5 * (dev[mid - 1] + dev[mid])
        } else {
            dev[mid]
        };
        // Winsorize: clamp each value to center ± tol, then average.
        let tol = winsor_k * mad + winsor_abs;
        let lo = center - tol;
        let hi = center + tol;
        let mut acc = 0.0f64;
        for &v in buf.iter() {
            acc += v.clamp(lo, hi);
        }
        *slot = acc * inv;
    }
    out
}

/// Loop-seam guard (reduction blend): pull the few samples at each loop
/// endpoint (the seam between index `n−1` and `0`) toward the band-limited
/// `master`. The blend weight `β` is a raised cosine — `1` exactly at a seam
/// sample, ramping to `0` at distance `radius`. Because the master is the
/// deterministic, **tick-free** reference, this attenuates a seam tick that is
/// *shared* across all engines (and so survives the median) by `(1−β)`, while
/// restoring exact loop continuity. The master is re-admitted ONLY in this
/// narrow guard — the body reduction still excludes it. No-op for
/// `radius == 0`.
fn apply_seam_master_blend(out: &mut [f64], master: &[f64], radius: usize) {
    let n = out.len().min(master.len());
    if radius == 0 || n < 3 {
        return;
    }
    // Cap so the two endpoint guards never overlap.
    let r = radius.min(n / 2);
    for k in 0..r {
        let beta = 0.5 * (1.0 + (std::f64::consts::PI * k as f64 / r as f64).cos());
        let head = k;
        out[head] = (1.0 - beta) * out[head] + beta * master[head];
        let tail = n - 1 - k;
        out[tail] = (1.0 - beta) * out[tail] + beta * master[tail];
    }
}

/// End-to-end 1D registration consensus with explicit [`LkParams`].
///
/// For each engine: compute LK flow master→engine, linearly warp the engine
/// onto the master grid, and push the warped engine. In **median** reduction
/// the (unwarped) master is *also* pushed as a self-regulating member (see
/// [`LkParams::include_master_in_reduction`]); in **mean**/**robust** mode it
/// is excluded. The stack is reduced per-sample by `mode`. `K == 1` returns the
/// lone engine unchanged (no warp, no master mixing); `K == 0` returns empty.
/// Output length is the minimum engine length. Production code calls this with
/// the runtime winsor knobs; tests use a default-param `registration_consensus_1d`
/// helper.
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

    let mut refs: Vec<&[f64]> = warped.iter().map(|v| v.as_slice()).collect();
    // Master as a full MEDIAN member (self-regulating "guidance"): it votes in
    // the source band and is discarded as a low outlier in the AI-extended
    // highs. Median only — Mean and Robust exclude it (Mean cannot reject it,
    // Robust keeps the AI highs undiluted). The master is the warp grid, so it
    // is added UNWARPED. Skip when it can't cover the full output (else the
    // min-length reducer would truncate the result).
    if params.include_master_in_reduction
        && mode == ReductionMode::Median
        && master.len() >= out_len
        && out_len >= 3
    {
        refs.push(&master[..out_len]);
    }
    let mut reduced = match mode {
        ReductionMode::Median => median_per_sample(&refs),
        ReductionMode::Mean => mean_per_sample(&refs),
        ReductionMode::Robust => robust_mean_per_sample(&refs, params.winsor_k, params.winsor_abs),
    };
    // Loop-seam guard (reduction blend): pull the few samples at each loop
    // endpoint toward the band-limited, tick-free master to remove a seam tick
    // shared across all engines and restore exact loop continuity. Body samples
    // are untouched. Inert unless looped and the radius is set.
    if params.seam_guard_radius > 0 && looped && align_n >= 3 {
        apply_seam_master_blend(&mut reduced, &master[..align_n], params.seam_guard_radius);
    }
    reduced
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience wrapper for the tests: the registration consensus with
    /// default LK params. Production code calls `registration_consensus_1d_with`
    /// directly so it can thread the runtime winsor knobs.
    fn registration_consensus_1d(
        master: &[f64],
        engines: &[&[f64]],
        looped: bool,
        mode: ReductionMode,
    ) -> Vec<f64> {
        registration_consensus_1d_with(master, engines, looped, mode, LkParams::default())
    }

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
    fn robust_mean_rejects_spike_keeps_mean_on_agreement() {
        // 5 engines: four agree at 0.30, one spikes to 0.95 at a few indices.
        // The robust mean must (a) reject the spike (≈ the agreeing value, like
        // a median) and (b) equal the PLAIN mean where all five agree.
        let n = 64;
        let agree = 0.30f64;
        let base = vec![agree; n];
        let mut spiky = base.clone();
        let spikes = [11usize, 23, 40];
        for &p in &spikes {
            spiky[p] = 0.95;
        }
        let refs = [
            base.as_slice(),
            base.as_slice(),
            base.as_slice(),
            base.as_slice(),
            spiky.as_slice(),
        ];
        let out = robust_mean_per_sample(&refs, 2.5, 0.01);
        // At a spike: median = 0.30, MAD = 0 (four agree), tol = floor = 0.01,
        // the 0.95 clamps to 0.31 ⇒ mean = (4·0.30 + 0.31)/5 = 0.302. The spike
        // leaks only floor/K — ≈ rejected (a plain mean would give 0.43).
        let plain_at_spike = (4.0 * agree + 0.95) / 5.0;
        for &p in &spikes {
            assert!(
                (out[p] - 0.302).abs() < 1e-9,
                "spike must clamp to median±floor, got {}",
                out[p]
            );
            assert!(
                (out[p] - plain_at_spike).abs() > 0.1,
                "robust must not behave like the plain mean at the spike"
            );
        }
        // Where all five agree exactly: every value inside the window ⇒ output
        // is the plain mean (== the agreed value).
        assert!(
            (out[0] - agree).abs() < 1e-12,
            "agreement must pass through as the mean"
        );
    }

    #[test]
    fn robust_mean_preserves_mean_under_benign_jitter() {
        // No outlier, only sub-window jitter ⇒ nothing clamps ⇒ the robust mean
        // equals the arithmetic mean (the σ/√N-smoothing contract), NOT the
        // median — this is what distinguishes it from `median_per_sample`.
        let vals = [0.20f64, 0.205, 0.198, 0.202];
        let cols: Vec<Vec<f64>> = vals.iter().map(|&v| vec![v; 8]).collect();
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        let out = robust_mean_per_sample(&refs, 2.5, 0.01);
        let plain = vals.iter().sum::<f64>() / 4.0;
        assert!(
            (out[0] - plain).abs() < 1e-9,
            "benign jitter must average like the mean, got {} vs {plain}",
            out[0]
        );
    }

    #[test]
    fn robust_mean_k0_fixed_band_and_k_large_is_plain_mean() {
        // Spread fixture (MAD > 0) so the k knob actually moves the window.
        let vals = [0.1f64, 0.2, 0.3, 0.4, 0.95];
        let cols: Vec<Vec<f64>> = vals.iter().map(|&v| vec![v; 4]).collect();
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        // k → ∞, floor 0 ⇒ no clamp ⇒ the plain mean (0.39).
        let big = robust_mean_per_sample(&refs, 1.0e6, 0.0);
        let plain = vals.iter().sum::<f64>() / 5.0;
        assert!((big[0] - plain).abs() < 1e-6, "k→∞ ⇒ plain mean, got {}", big[0]);
        // k = 0, floor 0 ⇒ clamp every value to the median ⇒ ≈ the median (0.3).
        let zero = robust_mean_per_sample(&refs, 0.0, 0.0);
        assert!((zero[0] - 0.3).abs() < 1e-9, "k=0,floor=0 ⇒ median, got {}", zero[0]);
    }

    #[test]
    fn robust_mean_is_sign_agnostic_on_negative_halfwave() {
        // The whole point vs the 2D brightness window: a spike on the NEGATIVE
        // half-wave is rejected identically to the positive (median.max(0) would
        // have zeroed the window here and let it through). center = -0.30.
        let n = 8;
        let base = vec![-0.30f64; n];
        let mut spiky = base.clone();
        spiky[3] = -0.95;
        let refs = [
            base.as_slice(),
            base.as_slice(),
            base.as_slice(),
            base.as_slice(),
            spiky.as_slice(),
        ];
        let out = robust_mean_per_sample(&refs, 2.5, 0.01);
        // tol = floor = 0.01; -0.95 clamps to -0.31 ⇒ (4·-0.30 + -0.31)/5 = -0.302.
        assert!(
            (out[3] + 0.302).abs() < 1e-9,
            "negative-half spike must clamp too, got {}",
            out[3]
        );
    }

    #[test]
    fn robust_mean_k1_passthrough_k0_empty() {
        let one = vec![0.1f64, -0.2, 0.3];
        assert_eq!(
            robust_mean_per_sample(&[one.as_slice()], 2.5, 0.01),
            one,
            "K=1 must pass the lone engine through unchanged"
        );
        let empty: [&[f64]; 0] = [];
        assert!(
            robust_mean_per_sample(&empty, 2.5, 0.01).is_empty(),
            "K=0 must return empty"
        );
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

    /// Band-limited square wave: sum of odd harmonics up to Nyquist, built as
    /// the **even** square (cosine series with alternating signs) so the sample
    /// at index 0 sits mid-plateau (value `+amp`) rather than on an edge. With
    /// `freq = cycles * SR / n` (integer `cycles`) it is periodic in `n`, so the
    /// loop seam is continuous and lands in the middle of a plateau. Carries the
    /// usual ~9% Gibbs overshoot at each edge — the chiptune case.
    fn band_limited_square(sr: f64, n: usize, freq: f64, amp: f64) -> Vec<f64> {
        let mut out = vec![0.0f64; n];
        let mut k = 1.0f64;
        let mut sign = 1.0f64;
        while k * freq < sr / 2.0 {
            let h = amp * 4.0 / std::f64::consts::PI / k * sign;
            for (i, s) in out.iter_mut().enumerate() {
                *s += h * (2.0 * std::f64::consts::PI * k * freq * i as f64 / sr).cos();
            }
            sign = -sign;
            k += 2.0;
        }
        out
    }

    #[test]
    fn chiptune_square_loop_keeps_edge_and_attenuates_seam_tick() {
        // Headline test. A looped band-limited square (the seam lands mid-
        // plateau, so it is continuous). Three engines are sub-sample shifts
        // {0, +0.3, +0.7} of it, each carrying the SAME tick injected at the
        // loop endpoints — a shared tick survives the per-sample median. The
        // seam guard must (a) keep the square's edges sharp in the interior and
        // (b) attenuate the shared seam tick; with the guard off the tick
        // survives (negative control).
        let n = 2048;
        let freq = 16.0 * SR / n as f64; // 16 cycles → continuous seam
        let master = band_limited_square(SR, n, freq, 0.8);
        let in_peak = master.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        let tick = 0.5 * in_peak;

        let make_engine = |s: f64| {
            let mut e = shift(&master, s, true);
            let last = e.len() - 1;
            e[0] += tick;
            e[1] += 0.5 * tick;
            e[last] += tick;
            e[last - 1] += 0.5 * tick;
            e
        };
        let e0 = make_engine(0.0);
        let e1 = make_engine(0.3);
        let e2 = make_engine(0.7);
        let refs = [e0.as_slice(), e1.as_slice(), e2.as_slice()];

        let guard = LkParams {
            sigma_a: 0.6,
            seam_guard_radius: 8,
            ..LkParams::default()
        };
        let no_guard = LkParams {
            sigma_a: 0.6,
            seam_guard_radius: 0,
            ..LkParams::default()
        };
        let out_g =
            registration_consensus_1d_with(&master, &refs, true, ReductionMode::Median, guard);
        let out_n =
            registration_consensus_1d_with(&master, &refs, true, ReductionMode::Median, no_guard);

        // (a) Interior (away from the seam guard): the aligned median tracks the
        // band-limited square — edges are not smeared and no extra overshoot is
        // introduced beyond the square's own Gibbs lobes.
        let margin = 64;
        let interior_peak = out_g[margin..n - margin]
            .iter()
            .fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(
            interior_peak <= in_peak * 1.05,
            "interior overshoot {interior_peak} exceeds in_peak {in_peak}"
        );
        let interior_diff = mean_abs_diff(&out_g[margin..n - margin], &master[margin..n - margin]);
        assert!(
            interior_diff < 0.1 * rms(&master),
            "interior should track the square (edges not smeared), diff={interior_diff}"
        );

        // (b) Seam-region deviation from the (tick-free) master, over the guard.
        let seam_dev = |out: &[f64]| {
            let mut d = 0.0f64;
            for k in 0..8usize {
                d = d.max((out[k] - master[k]).abs());
                d = d.max((out[n - 1 - k] - master[n - 1 - k]).abs());
            }
            d
        };
        let dev_g = seam_dev(&out_g);
        let dev_n = seam_dev(&out_n);
        assert!(
            dev_g < 0.15 * in_peak,
            "seam tick should be attenuated with the guard, dev={dev_g}"
        );
        assert!(
            dev_g < 0.5 * dev_n,
            "guard must beat no-guard at the seam: guard={dev_g}, no_guard={dev_n}"
        );
        // Negative control: the shared tick survives the median without a guard.
        assert!(
            dev_n > 0.4 * tick,
            "shared seam tick should survive without the guard, dev={dev_n}"
        );
    }

    #[test]
    fn bilateral_scale2_tracks_local_amplitude() {
        // The bilateral normaliser ≈ A² across a ±A square edge, ≈ 0 inside a
        // flat plateau, and exactly 0 in silence.
        let n = 512;
        let a = 0.7;
        let sq: Vec<f64> = (0..n)
            .map(|i| if (i / 40) % 2 == 0 { a } else { -a })
            .collect();
        let r = 5isize;
        let v_flat = bilateral_scale2(&sq, 20, r, n, false);
        assert!(
            v_flat < 1e-9,
            "variance inside a flat plateau should be ~0, got {v_flat}"
        );
        let v_edge = bilateral_scale2(&sq, 40, r, n, false);
        assert!(
            (v_edge - a * a).abs() < 0.2 * a * a,
            "variance across an edge should be ≈ A², got {v_edge}"
        );
        let silence = vec![0.0f64; n];
        assert_eq!(bilateral_scale2(&silence, 100, r, n, false), 0.0);
    }

    #[test]
    fn bilateral_weight_changes_flow_on_square_edges() {
        // The bilateral weight must actually engage: on a square's edges it
        // produces a different flow field than pure spatial (uniform)
        // weighting. (Proves the feature is wired in, independent of the
        // library default.)
        let n = 1024;
        let freq = 6.0 * SR / n as f64;
        let master = band_limited_square(SR, n, freq, 0.8);
        let engine = shift(&master, 0.45, false);
        let f_bilat = dense_lk_1d(
            &master,
            &engine,
            false,
            LkParams {
                sigma_a: 0.6,
                ..LkParams::default()
            },
        );
        let f_unif = dense_lk_1d(
            &master,
            &engine,
            false,
            LkParams {
                sigma_a: f64::INFINITY,
                ..LkParams::default()
            },
        );
        let max_diff = f_bilat
            .iter()
            .zip(&f_unif)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1e-3,
            "bilateral weight must change the flow vs uniform on square edges, max_diff={max_diff}"
        );
    }

    #[test]
    fn bilateral_weight_is_level_invariant() {
        // A loud and a quiet square (same shape, ×100 amplitude) must register
        // to the same flow — the window-local variance normaliser cancels the
        // amplitude. (Guards against a mis-scaled bilateral that would weight by
        // absolute amplitude.)
        let n = 1024;
        let freq = 8.0 * SR / n as f64;
        let loud = band_limited_square(SR, n, freq, 0.9);
        let quiet: Vec<f64> = loud.iter().map(|x| x * 0.01).collect();
        let p = LkParams {
            sigma_a: 0.6,
            ..LkParams::default()
        };
        let f_loud = dense_lk_1d(&loud, &shift(&loud, 0.4, false), false, p);
        let f_quiet = dense_lk_1d(&quiet, &shift(&quiet, 0.4, false), false, p);
        let margin = 32;
        let mean = |f: &[f64]| f[margin..n - margin].iter().sum::<f64>() / (n - 2 * margin) as f64;
        assert!(
            (mean(&f_loud) - mean(&f_quiet)).abs() < 0.05,
            "loud and quiet squares must register identically: loud={}, quiet={}",
            mean(&f_loud),
            mean(&f_quiet)
        );
    }

    #[test]
    fn bilateral_weight_ignores_gibbs_ripple() {
        // A band-limited square (rich Gibbs lobes) registered against itself
        // must yield ~zero flow — the ripple does not manufacture spurious
        // motion through the bilateral weight.
        let n = 1024;
        let freq = 10.0 * SR / n as f64;
        let master = band_limited_square(SR, n, freq, 0.8);
        let flow = dense_lk_1d(
            &master,
            &master,
            false,
            LkParams {
                sigma_a: 0.6,
                ..LkParams::default()
            },
        );
        let max_u = flow.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(max_u < 1e-6, "identity registration should give ~zero flow, max_u={max_u}");
    }

    #[test]
    fn seam_guard_is_inert_when_not_looped() {
        // Edge-aware weighting and the seam guard are both ON by default.
        let d = LkParams::default();
        assert!(
            d.sigma_a.is_finite() && d.sigma_a > 0.0,
            "edge-aware weighting must be on by default"
        );
        assert!(d.seam_guard_radius > 0, "seam guard must be on by default");
        // The seam guard only acts on looped buffers; with looped=false it must
        // be a no-op for any radius (no master re-admission off-loop).
        let n = 512;
        let master = tone(SR, n, 600.0, 0.6, 0.0);
        let e1 = shift(&master, 0.2, false);
        let e2 = shift(&master, 0.5, false);
        let refs = [e1.as_slice(), e2.as_slice()];
        let off = LkParams {
            seam_guard_radius: 0,
            ..LkParams::default()
        };
        let on = LkParams {
            seam_guard_radius: 8,
            ..LkParams::default()
        };
        let a = registration_consensus_1d_with(&master, &refs, false, ReductionMode::Median, off);
        let b = registration_consensus_1d_with(&master, &refs, false, ReductionMode::Median, on);
        assert_eq!(a, b, "seam guard must be inert when looped=false");
    }

    #[test]
    fn high_pitch_square_no_panic() {
        // Square whose period is shorter than the window (edges within the
        // window): the bilateral weight may cut nearly every tap, but the centre
        // tap (w_photo=1) and the scale eps keep the solve defined. Must not
        // panic, NaN, or blow up.
        let n = 600;
        let freq = SR / 8.0; // period 8 samples
        let master = band_limited_square(SR, n, freq, 0.8);
        let e1 = shift(&master, 0.3, true);
        let e2 = shift(&master, 0.6, true);
        let refs = [e1.as_slice(), e2.as_slice()];
        let p = LkParams {
            sigma_a: 0.6,
            seam_guard_radius: 8,
            ..LkParams::default()
        };
        let out = registration_consensus_1d_with(&master, &refs, true, ReductionMode::Median, p);
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|x| x.is_finite()), "output must be finite");
        let in_peak = master.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        let out_peak = out.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(
            out_peak <= in_peak * 1.5,
            "output should stay bounded, out={out_peak} in={in_peak}"
        );
    }

    #[test]
    fn include_master_in_reduction_on_by_default() {
        assert!(
            LkParams::default().include_master_in_reduction,
            "master guidance must be on by default"
        );
    }

    #[test]
    fn master_member_preserves_highs_in_median() {
        // The master is band-limited (220 Hz only); the engines carry an extra
        // 6 kHz the master lacks. With the master a MEDIAN member, it is a low
        // outlier in the 6 kHz band — the median discards it — so the AI high
        // band survives in full, while the shared in-band content is anchored.
        let n = 8192;
        let master = tone(SR, n, 220.0, 0.5, 0.0);
        let engine: Vec<f64> = (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / SR).sin()
                    + 0.4 * (2.0 * std::f64::consts::PI * 6000.0 * i as f64 / SR).sin()
            })
            .collect();
        let refs = [engine.as_slice(), engine.as_slice()];
        let out = registration_consensus_1d(&master, &refs, false, ReductionMode::Median);

        let amp_engine = component_amplitude(&engine, 6000.0, SR);
        let amp_out = component_amplitude(&out, 6000.0, SR);
        assert!(
            amp_out > 0.9 * amp_engine,
            "median rejects the band-limited master in the high band; 6 kHz must \
             survive: out={amp_out:.4}, engine={amp_engine:.4}"
        );
        assert!(
            mean_abs_diff(&out, &engine) < 0.05 * rms(&engine),
            "median output should track the engines on shared content"
        );
    }

    #[test]
    fn master_member_rejects_inband_hallucination() {
        // Even split: two clean engines (== master) and two that share an in-band
        // 660 Hz hallucination. Engines alone (median of 4) average a clean and a
        // dirty middle, leaking the hallucination; the master's correct vote
        // breaks the tie (median of 5 → the clean value).
        let n = 8192;
        let master = tone(SR, n, 440.0, 0.5, 0.0);
        let clean = master.clone();
        let dirty: Vec<f64> = (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 660.0 * i as f64 / SR).sin()
            })
            .collect();
        let refs = [
            clean.as_slice(),
            clean.as_slice(),
            dirty.as_slice(),
            dirty.as_slice(),
        ];
        let on = LkParams::default(); // include_master_in_reduction = true
        let off = LkParams {
            include_master_in_reduction: false,
            ..LkParams::default()
        };
        let out_on =
            registration_consensus_1d_with(&master, &refs, false, ReductionMode::Median, on);
        let out_off =
            registration_consensus_1d_with(&master, &refs, false, ReductionMode::Median, off);

        let leak_on = component_amplitude(&out_on, 660.0, SR);
        let leak_off = component_amplitude(&out_off, 660.0, SR);
        assert!(
            leak_on < 0.5 * leak_off,
            "the master's vote should break the even-split tie: on={leak_on:.4}, \
             off={leak_off:.4}"
        );
        assert!(
            leak_on < 0.1,
            "with the master, the 660 Hz hallucination is rejected: {leak_on:.4}"
        );
    }

    #[test]
    fn master_excluded_from_mean_even_when_flag_on() {
        // The median-only gate: in MEAN mode the master is never a member, even
        // with the flag on, so the AI high band is not diluted by its zero highs.
        let n = 8192;
        let master = tone(SR, n, 220.0, 0.5, 0.0);
        let engine: Vec<f64> = (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / SR).sin()
                    + 0.4 * (2.0 * std::f64::consts::PI * 6000.0 * i as f64 / SR).sin()
            })
            .collect();
        let refs = [engine.as_slice(), engine.as_slice()];
        let p = LkParams {
            include_master_in_reduction: true,
            ..LkParams::default()
        };
        let out = registration_consensus_1d_with(&master, &refs, false, ReductionMode::Mean, p);
        let amp_engine = component_amplitude(&engine, 6000.0, SR);
        let amp_out = component_amplitude(&out, 6000.0, SR);
        assert!(
            amp_out > 0.9 * amp_engine,
            "mean must keep the master excluded even with the flag on (no HF \
             dilution): out={amp_out:.4}, engine={amp_engine:.4}"
        );
    }
}
