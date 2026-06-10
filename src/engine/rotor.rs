// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

// Rotor-aware operations on phasors.
//
// A complex number z = r·exp(iθ) is structurally a 2D rotor in Cl⁺(2): a
// magnitude r ∈ ℝ⁺ paired with a unit rotor exp(iθ) ∈ S¹. These factors
// live on different manifolds — magnitudes on the multiplicative group
// (ℝ⁺, ·), phases on the circle S¹ — and operations on phasors should
// respect that structure.
//
// Linear interpolation in the Cartesian (ℝ²) embedding does not. When two
// phasors of similar magnitude have disagreeing phases, the chord between
// them in ℂ passes closer to the origin than either endpoint — i.e. the
// blend silently attenuates the magnitude based on phase mismatch. In an
// inverse STFT, that per-bin attenuation manifests as pre-echo,
// transient smearing, and comb-filter coloration in the time domain.
//
// The rotor-correct interpolation is proper SLERP, which is the geodesic
// on the multiplicative group of nonzero complex numbers ℂ\{0} ≅ ℝ⁺ × S¹:
//
//     SLERP(R₁, R₂, t) = R₁ · (R₁⁻¹R₂)^t
//                      = exp((1-t) log r₁ + t log r₂)
//                        · exp(i(θ₁ + t · shortest_arc(θ₁, θ₂)))
//
// — geometric mean on the magnitude axis, additive shortest-arc on the
// phase axis. Endpoints land on R₁ and R₂ exactly. At intermediate t,
// the magnitude no longer depends on phase disagreement.

#![allow(dead_code)]

use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Magnitudes below this are treated as zero — phases are undefined and
/// the magnitude geodesic on (ℝ⁺, ·) is not reachable from the boundary.
pub const MAGNITUDE_FLOOR: f64 = 1e-30;

/// Signed shortest angular distance from `from` to `to`, normalized to
/// (-π, π]. Exact antipodes (|difference| ≡ π mod 2π) return +π.
///
/// Equivalent to `arg(exp(i·to) / exp(i·from))` taken on the principal
/// branch — i.e. the Lie-algebra element of the relative S¹ rotation
/// `R_from⁻¹ · R_to`.
pub fn shortest_arc(from: f64, to: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut d = (to - from) % two_pi;
    if d > PI {
        d -= two_pi;
    } else if d <= -PI {
        d += two_pi;
    }
    d
}

/// Rotor SLERP between two phasors: geodesic interpolation on ℝ⁺ × S¹.
///
/// - At `t = 0` returns `a`; at `t = 1` returns `b` (within numerical
///   precision).
/// - Magnitude is the geometric mean: `|a|^(1-t) · |b|^t`.
/// - Phase is additive along the shorter arc on S¹.
///
/// **Boundary handling.** When either magnitude is below `MAGNITUDE_FLOOR`
/// we drop to the boundary of (ℝ⁺, ·): magnitude falls back to a linear
/// lerp (the geodesic doesn't reach 0 in finite parameter), and the phase
/// is taken from whichever side has signal. When both are zero, the
/// result is zero.
///
/// **Antipodal handling.** When the phases differ by exactly ±π the
/// shorter arc is undefined (two equivalent geodesics). We fall back to a
/// Cartesian blend at that bin only — this preserves the Hermitian
/// symmetry that real-input FFTs rely on, at the cost of magnitude
/// attenuation in that one bin (which is unavoidable: a 180° phase flip
/// really is destructive interference).
pub fn polar_slerp(a: Complex<f64>, b: Complex<f64>, t: f64) -> Complex<f64> {
    // Interpolation only — extrapolation through the magnitude-floor branches
    // below would produce a negative radius, which `from_polar` renders as a
    // silent pi phase flip.
    let t = t.clamp(0.0, 1.0);
    let r_a = a.norm();
    let r_b = b.norm();

    if r_a < MAGNITUDE_FLOOR && r_b < MAGNITUDE_FLOOR {
        return Complex::new(0.0, 0.0);
    }

    if r_a < MAGNITUDE_FLOOR {
        return Complex::from_polar(r_a * (1.0 - t) + r_b * t, b.arg());
    }
    if r_b < MAGNITUDE_FLOOR {
        return Complex::from_polar(r_a * (1.0 - t) + r_b * t, a.arg());
    }

    let theta_a = a.arg();
    let theta_b = b.arg();
    let arc = shortest_arc(theta_a, theta_b);

    // Antipodal: shortest arc has two equally-valid directions. Cartesian
    // blend collapses to zero at t=0.5 there, but that's the physically
    // correct answer for two equally-loud signals in counter-phase.
    const ANTIPODE_EPS: f64 = 1e-9;
    if (arc.abs() - PI).abs() < ANTIPODE_EPS {
        return a * (1.0 - t) + b * t;
    }

    let r = (r_a.ln() * (1.0 - t) + r_b.ln() * t).exp();
    let theta = theta_a + t * arc;
    Complex::from_polar(r, theta)
}

/// Phase-aware lerp between two phasors: arithmetic-mean magnitude on
/// `ℝ⁺`, magnitude-weighted phase on `S¹`.
///
/// This is the right primitive for **crossfades** between two signals
/// that represent the same content at different processing stages (e.g.
/// LR source vs upsampled candidate). The phase comes from the Cartesian
/// blend `(1-t)·a + t·b`, which has three useful properties simultaneously:
///
/// 1. When magnitudes are equal and phases agree, it returns the same
///    phase as SLERP would — the geodesic midpoint on `S¹`.
/// 2. When one magnitude dominates the other (e.g. one side is
///    windowing-leakage at a bin where the other side carries the actual
///    signal), the phase is dominated by the louder side. SLERP'ing
///    halfway toward random sidelobe phase would smear the dominant
///    signal; this avoids that.
/// 3. The output magnitude is `r_a·(1-t) + r_b·t` (taken as a separate
///    arithmetic lerp, *not* the shortened Cartesian-chord magnitude),
///    so phase disagreement at equal magnitudes does NOT silently shrink
///    the bin's energy — that's the bug we're fixing relative to a naive
///    `a·(1-t) + b·t` Cartesian blend.
///
/// Hermitian symmetry holds: `polar_lerp(conj(a), conj(b), t) ==
/// conj(polar_lerp(a, b, t))`, since both the Cartesian-blend phase and
/// the magnitude lerp are odd / even (respectively) under conjugation.
///
/// Compare to [`polar_slerp`], which uses geometric-mean magnitude — the
/// Karcher geodesic on `(ℝ⁺, ·)`. The geometric variant is correct for
/// **consensus** of N independent estimates (biased toward the smallest
/// input) but is too aggressive for a crossfade where one side may carry
/// no signal at a bin: geometric mean of `(r, 0+ε)` collapses to ≈ 0 and
/// kills the dominant side.
pub fn polar_lerp(a: Complex<f64>, b: Complex<f64>, t: f64) -> Complex<f64> {
    // Interpolation only — see `polar_slerp`.
    let t = t.clamp(0.0, 1.0);
    let r_a = a.norm();
    let r_b = b.norm();

    if r_a < MAGNITUDE_FLOOR && r_b < MAGNITUDE_FLOOR {
        return Complex::new(0.0, 0.0);
    }

    let r = r_a * (1.0 - t) + r_b * t;

    let cartesian = a * (1.0 - t) + b * t;
    let theta = if cartesian.norm() < MAGNITUDE_FLOOR {
        // Antipodal phases at near-equal magnitudes — the Cartesian blend
        // collapses to ≈ 0 and arg() is meaningless. Use the dominant
        // side's phase (or `a`'s if exactly equal).
        if r_a >= r_b { a.arg() } else { b.arg() }
    } else {
        cartesian.arg()
    };

    Complex::from_polar(r, theta)
}

/// Geometric mean of magnitudes — the Karcher mean on the multiplicative
/// group `(ℝ⁺, ·)`.
///
/// `(∏ rᵢ)^(1/N)`, computed via log-sum-exp to avoid overflow/underflow on
/// long products. This is the rotor-correct magnitude consensus across N
/// engines: smooth in every input, biased toward smaller magnitudes (a
/// single very-small engine pulls the output strongly toward 0, preserving
/// the "trust the quietest engine" intent of softmin without patching
/// together pieces of different engines' spectra at different bins).
///
/// Magnitudes below `MAGNITUDE_FLOOR` are clamped up to it for the log
/// computation — they still drag the output toward zero (since
/// `MAGNITUDE_FLOOR^(1/N)` is small) but don't produce `-inf`.
pub fn geometric_mean_magnitude(magnitudes: &[f64]) -> f64 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    if magnitudes.len() == 1 {
        return magnitudes[0];
    }
    let n = magnitudes.len() as f64;
    let log_sum: f64 = magnitudes
        .iter()
        .map(|&m| m.max(MAGNITUDE_FLOOR).ln())
        .sum();
    (log_sum / n).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn complex_close(a: Complex<f64>, b: Complex<f64>, eps: f64) -> bool {
        (a - b).norm() < eps
    }

    #[test]
    fn shortest_arc_simple_directions() {
        assert!(approx(shortest_arc(0.0, PI / 2.0), PI / 2.0, 1e-12));
        assert!(approx(shortest_arc(PI / 2.0, 0.0), -PI / 2.0, 1e-12));
        assert!(approx(shortest_arc(0.0, 0.0), 0.0, 1e-12));
    }

    #[test]
    fn shortest_arc_takes_shorter_route() {
        let a = shortest_arc(0.1, 6.0);
        assert!(a < 0.0 && a > -PI, "expected small negative, got {a}");
    }

    #[test]
    fn shortest_arc_inputs_outside_principal_branch() {
        let a = shortest_arc(10.0 * PI + 0.1, 10.0 * PI + 0.3);
        assert!(approx(a, 0.2, 1e-9));
    }

    #[test]
    fn polar_slerp_endpoints() {
        let a = Complex::from_polar(2.0, 0.5);
        let b = Complex::from_polar(3.0, 1.5);
        assert!(complex_close(polar_slerp(a, b, 0.0), a, 1e-12));
        assert!(complex_close(polar_slerp(a, b, 1.0), b, 1e-12));
    }

    #[test]
    fn polar_slerp_does_not_attenuate_at_phase_disagreement() {
        // Two unit phasors at 90° apart. Cartesian lerp at t=0.5 has magnitude
        // sqrt(2)/2 ≈ 0.707 — that's the bug. Rotor SLERP keeps magnitude 1.
        let a = Complex::from_polar(1.0, 0.0);
        let b = Complex::from_polar(1.0, PI / 2.0);
        let mid = polar_slerp(a, b, 0.5);
        assert!(
            approx(mid.norm(), 1.0, 1e-12),
            "magnitude was {}",
            mid.norm()
        );
        assert!(approx(mid.arg(), PI / 4.0, 1e-12));

        let cartesian = a * 0.5 + b * 0.5;
        assert!(cartesian.norm() < 0.8, "Cartesian comparison sanity");
    }

    #[test]
    fn polar_slerp_geometric_mean_magnitude() {
        let a = Complex::from_polar(1.0, 0.3);
        let b = Complex::from_polar(4.0, 0.3);
        let mid = polar_slerp(a, b, 0.5);
        let expected = (1.0_f64 * 4.0).sqrt(); // = 2.0
        assert!(
            approx(mid.norm(), expected, 1e-9),
            "magnitude was {}",
            mid.norm()
        );
        assert!(approx(mid.arg(), 0.3, 1e-9));
    }

    #[test]
    fn polar_slerp_preserves_hermitian_symmetry() {
        let cases = [
            (Complex::from_polar(2.0, 0.7), Complex::from_polar(3.0, 1.2)),
            (
                Complex::from_polar(0.5, -1.1),
                Complex::from_polar(1.5, 0.4),
            ),
            (
                Complex::from_polar(1.0, PI - 0.01),
                Complex::from_polar(1.0, -PI + 0.01),
            ),
        ];
        for (a, b) in cases {
            for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let result = polar_slerp(a, b, t);
                let result_conj = polar_slerp(a.conj(), b.conj(), t);
                assert!(
                    complex_close(result.conj(), result_conj, 1e-10),
                    "Hermitian broken at t={t} for ({a}, {b}): result.conj()={}, expected={}",
                    result.conj(),
                    result_conj
                );
            }
        }
    }

    #[test]
    fn polar_slerp_zero_input_uses_other_phase() {
        let zero = Complex::new(0.0, 0.0);
        let b = Complex::from_polar(1.0, 0.7);
        let mid = polar_slerp(zero, b, 0.5);
        assert!(approx(mid.norm(), 0.5, 1e-12));
        assert!(approx(mid.arg(), 0.7, 1e-12));

        let mid2 = polar_slerp(b, zero, 0.5);
        assert!(approx(mid2.norm(), 0.5, 1e-12));
        assert!(approx(mid2.arg(), 0.7, 1e-12));
    }

    #[test]
    fn polar_slerp_both_zero_is_zero() {
        let zero = Complex::new(0.0, 0.0);
        assert!(polar_slerp(zero, zero, 0.5).norm() < 1e-15);
    }

    #[test]
    fn polar_slerp_antipodal_falls_back_to_cartesian() {
        let a = Complex::from_polar(1.0, 0.0);
        let b = Complex::from_polar(1.0, PI);
        let mid = polar_slerp(a, b, 0.5);
        assert!(mid.norm() < 1e-12, "expected ≈0, got {}", mid.norm());
    }

    #[test]
    fn geometric_mean_single_value() {
        assert!((geometric_mean_magnitude(&[0.5]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn geometric_mean_equal_values() {
        let result = geometric_mean_magnitude(&[0.3, 0.3, 0.3]);
        assert!((result - 0.3).abs() < 1e-9, "Got {result}");
    }

    #[test]
    fn geometric_mean_biased_toward_smallest() {
        let result = geometric_mean_magnitude(&[1.0, 0.001]);
        let arithmetic = (1.0 + 0.001) / 2.0;
        assert!(
            result < arithmetic,
            "geom {} < arith {}",
            result,
            arithmetic
        );
        assert!(result > 0.001, "geom {} > min {}", result, 0.001);
        assert!((result - 0.001_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn polar_lerp_arithmetic_magnitude_at_midpoint() {
        // Crossfade contract: at t=0.5, magnitude is the arithmetic mean of
        // the endpoints' magnitudes. This is what an audio crossfade
        // expects (constant-energy mixing) and what differentiates
        // polar_lerp from polar_slerp.
        let a = Complex::from_polar(1.0, 0.3);
        let b = Complex::from_polar(0.001, 0.3);
        let mid = polar_lerp(a, b, 0.5);
        assert!(approx(mid.norm(), 0.5005, 1e-9));
    }

    #[test]
    fn polar_lerp_endpoints() {
        let a = Complex::from_polar(2.0, 0.5);
        let b = Complex::from_polar(3.0, 1.5);
        assert!(complex_close(polar_lerp(a, b, 0.0), a, 1e-12));
        assert!(complex_close(polar_lerp(a, b, 1.0), b, 1e-12));
    }

    #[test]
    fn polar_lerp_does_not_attenuate_at_phase_disagreement() {
        // Same property as polar_slerp: at 90° phase disagreement, the
        // magnitude of the midpoint stays at the arithmetic mean (= 1.0
        // for two unit phasors), not the Cartesian-chord-shortened value.
        let a = Complex::from_polar(1.0, 0.0);
        let b = Complex::from_polar(1.0, PI / 2.0);
        let mid = polar_lerp(a, b, 0.5);
        assert!(
            approx(mid.norm(), 1.0, 1e-12),
            "magnitude was {}",
            mid.norm()
        );
        assert!(approx(mid.arg(), PI / 4.0, 1e-12));
    }

    #[test]
    fn polar_lerp_preserves_hermitian_symmetry() {
        let cases = [
            (Complex::from_polar(2.0, 0.7), Complex::from_polar(3.0, 1.2)),
            (
                Complex::from_polar(0.5, -1.1),
                Complex::from_polar(1.5, 0.4),
            ),
        ];
        for (a, b) in cases {
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let result = polar_lerp(a, b, t);
                let result_conj = polar_lerp(a.conj(), b.conj(), t);
                assert!(complex_close(result.conj(), result_conj, 1e-10));
            }
        }
    }

    #[test]
    fn geometric_mean_smooth_in_inputs() {
        let base = [0.5, 0.7];
        let perturbed = [0.5, 0.7 + 1e-6];
        let g_base = geometric_mean_magnitude(&base);
        let g_pert = geometric_mean_magnitude(&perturbed);
        assert!(
            (g_base - g_pert).abs() < 1e-5,
            "not smooth: {g_base} vs {g_pert}"
        );
    }
}
