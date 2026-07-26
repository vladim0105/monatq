use std::sync::OnceLock;

use statrs::distribution::{ContinuousCDF, Laplace, LogNormal, Normal, Uniform};
use strum::IntoEnumIterator;
use wide::f32x8;

pub(crate) const N_PROBES: usize = 50;
/// Padded to the next multiple of 8 so the SIMD loop in `kernel` needs no scalar tail.
pub(crate) const N_PADDED: usize = 56;
/// Number of known (non-Unknown) distribution variants.
pub(crate) const N_DISTRIBUTIONS: usize = 5;

/// Evenly-spaced probe quantiles: p_i = (i + 0.5) / N_PROBES.
pub(crate) fn probe_points() -> [f32; N_PROBES] {
    std::array::from_fn(|i| (i as f32 + 0.5) / N_PROBES as f32)
}

/// Normalised reference quantile profiles for each known distribution,
/// evaluated at the fixed probe points and zero-padded to `N_PADDED`.
/// Computed once on first use; reused for every `analyze()` call.
pub(crate) struct RefProfiles(pub(crate) [[f32; N_PADDED]; N_DISTRIBUTIONS]);

pub(crate) fn ref_profiles() -> &'static RefProfiles {
    static CACHE: OnceLock<RefProfiles> = OnceLock::new();
    CACHE.get_or_init(|| {
        debug_assert_eq!(
            Distribution::iter().count(),
            N_DISTRIBUTIONS,
            "N_DISTRIBUTIONS is out of sync with Distribution::iter()"
        );
        let probes = probe_points();
        let mut data = [[0f32; N_PADDED]; N_DISTRIBUTIONS];
        for d in Distribution::iter() {
            let ref_med = d.reference_quantile(0.5);
            let ref_std = (d.reference_quantile(0.84) - d.reference_quantile(0.16)) / 2.0;
            let slot = &mut data[d.index()];
            for i in 0..N_PROBES {
                slot[i] = (d.reference_quantile(probes[i]) - ref_med) / ref_std;
            }
            // positions N_PROBES..N_PADDED stay 0.0 — contribute nothing to the sum
        }
        RefProfiles(data)
    })
}

/// L1 distance above which no reference family is considered a match.
///
/// Calibrated against the t-digest kernel; shared so every kernel classifies on the same
/// scale and `analyze` results stay comparable across backends.
const UNKNOWN_THRESHOLD: f32 = 10.8;

/// Classify one empirical distribution from its quantile function.
///
/// `quantile` is the only thing a kernel has to supply, which is why this lives here rather
/// than in a kernel: the procedure is identical for any summary that can answer a rank
/// query. Standardise the empirical profile by its own median and interquantile spread, then
/// take the nearest reference profile in L1.
///
/// Returns [`Distribution::Unknown`] for a degenerate summary (no measurable spread) or when
/// the best fit is still too far away.
pub(crate) fn classify(mut quantile: impl FnMut(f32) -> f32) -> Distribution {
    let median = quantile(0.5);
    let spread = (quantile(0.84) - quantile(0.16)) / 2.0;
    if !spread.is_finite() || spread.abs() < 1e-6 {
        return Distribution::Unknown;
    }

    let probes = probe_points();
    let mut empirical = [0f32; N_PADDED];
    for (slot, &p) in empirical.iter_mut().zip(probes.iter()) {
        *slot = (quantile(p) - median) / spread;
    }

    let profiles = ref_profiles();
    let (best, best_distance) = Distribution::iter()
        .map(|d| (d, l1_distance(&empirical, &profiles.0[d.index()])))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("Distribution::iter() is never empty");

    if best_distance > UNKNOWN_THRESHOLD {
        Distribution::Unknown
    } else {
        best
    }
}

/// Sum of absolute differences over `N_PADDED` lanes.
///
/// `N_PADDED` is a multiple of 8 and the tail is zero-filled, so this needs no scalar
/// remainder loop.
fn l1_distance(a: &[f32; N_PADDED], b: &[f32; N_PADDED]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    for i in (0..N_PADDED).step_by(8) {
        let va = f32x8::from(<[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        let diff = va - vb;
        acc += diff.max(-diff);
    }
    acc.reduce_add()
}

/// Distribution family identified by [`TensorDigest::analyze`](crate::TensorDigest::analyze).
///
/// Each variant corresponds to a canonical shape matched against the empirical
/// quantile profile via L1 distance. `Unknown` is returned when no family fits
/// within the calibrated threshold.
#[derive(Debug, Clone, Copy, PartialEq, strum::EnumIter, serde::Serialize, serde::Deserialize)]
pub enum Distribution {
    Normal,
    Uniform,
    Laplace,
    LogNormal,
    /// Symmetric mixture of two Gaussians: 0.5·N(-d, σ_c) + 0.5·N(+d, σ_c)
    /// with d = √3/2, σ_c = 0.5 (zero-mean, unit-variance).
    BiNormal,
    /// Best-fit L1 distance exceeded the calibrated threshold; distribution
    /// shape does not closely match any of the known families, or the data
    /// is degenerate (constant / zero variance).
    #[strum(disabled)]
    Unknown,
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Normal => "Normal",
            Self::Uniform => "Uniform",
            Self::Laplace => "Laplace",
            Self::LogNormal => "LogNormal",
            Self::BiNormal => "BiNormal",
            Self::Unknown => "Unknown",
        })
    }
}

impl Distribution {
    /// Index into the `RefProfiles` array. Only valid for known (non-Unknown) variants.
    pub(crate) fn index(self) -> usize {
        match self {
            Distribution::Normal => 0,
            Distribution::Uniform => 1,
            Distribution::Laplace => 2,
            Distribution::LogNormal => 3,
            Distribution::BiNormal => 4,
            Distribution::Unknown => unreachable!(),
        }
    }

    pub(crate) fn reference_quantile(self, p: f32) -> f32 {
        let p64 = p as f64;
        match self {
            Distribution::Normal => Normal::new(0.0, 1.0).unwrap().inverse_cdf(p64) as f32,
            // Uniform on [-√3, √3]: mean=0, variance=1
            Distribution::Uniform => Uniform::new(-(3f64.sqrt()), 3f64.sqrt())
                .unwrap()
                .inverse_cdf(p64) as f32,
            // Laplace(0, 1/√2): mean=0, variance=1
            Distribution::Laplace => Laplace::new(0.0, 1.0 / 2f64.sqrt())
                .unwrap()
                .inverse_cdf(p64) as f32,
            // LogNormal(0,1) standardized to mean=0, variance=1
            Distribution::LogNormal => {
                const MEAN: f32 = 1.6487213;
                const STD: f32 = 2.1612;
                (LogNormal::new(0.0, 1.0).unwrap().inverse_cdf(p64) as f32 - MEAN) / STD
            }
            // Symmetric bimodal: 0.5·N(-d, σ_c) + 0.5·N(+d, σ_c)
            // d = √3/2, σ_c = 0.5 → mean=0, variance = d² + σ_c² = 0.75 + 0.25 = 1
            // CDF(x) = 0.5·Φ((x+d)/σ_c) + 0.5·Φ((x-d)/σ_c)
            // Invert numerically via bisection.
            Distribution::BiNormal => {
                const D: f64 = 0.8660254037844387; // sqrt(3)/2
                const SC: f64 = 0.5;
                let n = Normal::new(0.0, 1.0).unwrap();
                let cdf = |x: f64| 0.5 * n.cdf((x + D) / SC) + 0.5 * n.cdf((x - D) / SC);
                let mut lo = -20f64;
                let mut hi = 20f64;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if cdf(mid) < p64 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                ((lo + hi) / 2.0) as f32
            }
            Distribution::Unknown => unreachable!("Unknown has no reference quantile"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_quantile_normal() {
        assert!((Distribution::Normal.reference_quantile(0.5) - 0.0).abs() < 1e-5);
        assert!((Distribution::Normal.reference_quantile(0.84) - 1.0).abs() < 0.01);
        assert!((Distribution::Normal.reference_quantile(0.16) - (-1.0)).abs() < 0.01);
        assert!((Distribution::Normal.reference_quantile(0.975) - 1.96).abs() < 0.01);
    }

    #[test]
    fn reference_quantile_uniform() {
        assert!((Distribution::Uniform.reference_quantile(0.5) - 0.0).abs() < 1e-5);
        assert!(
            (Distribution::Uniform.reference_quantile(0.75) - 0.5 * 3.0f32.sqrt()).abs() < 1e-5
        );
    }

    #[test]
    fn reference_quantile_laplace() {
        assert!((Distribution::Laplace.reference_quantile(0.5) - 0.0).abs() < 1e-5);
        let expected = -(2.0f32 * (1.0 - 0.84)).ln() / std::f32::consts::SQRT_2;
        assert!((Distribution::Laplace.reference_quantile(0.84) - expected).abs() < 1e-5);
    }

    #[test]
    fn reference_quantile_lognormal() {
        let expected = (1.0f32 - 1.6487213) / 2.1612;
        assert!((Distribution::LogNormal.reference_quantile(0.5) - expected).abs() < 1e-4);
    }
}
