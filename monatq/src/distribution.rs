use std::sync::OnceLock;

use statrs::distribution::{ContinuousCDF, Laplace, LogNormal, Normal, Uniform};
use strum::IntoEnumIterator;

pub(crate) const N_PROBES: usize = 50;
/// Padded to the next multiple of 8 so the SIMD loop in `kernel` needs no scalar tail.
pub(crate) const N_PADDED: usize = 56;
/// Number of known (non-Unknown) distribution variants.
pub(crate) const N_DISTRIBUTIONS: usize = 4;

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

/// Distribution family identified by [`TensorDigest::analyze`].
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
    /// Best-fit L1 distance exceeded the calibrated threshold; distribution
    /// shape does not closely match any of the four known families.
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
