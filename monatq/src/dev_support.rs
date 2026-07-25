//! Shared backend adapter for repository benchmarks and accuracy reports.

use crate::{QuantileSpine, TDigest, TensorDigest};

/// Digest implementations exercised by backend comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    TDigest,
    QuantileSpine,
}

/// The single backend list used by performance and accuracy reporting.
pub const BACKENDS: &[Backend] = &[Backend::TDigest, Backend::QuantileSpine];

/// Type-erased adapter used only by repository tooling.
pub enum Digest {
    TDigest(TensorDigest<f32, TDigest>),
    QuantileSpine(TensorDigest<f32, QuantileSpine>),
}

impl Backend {
    pub const BASELINE: Self = Self::TDigest;

    pub fn create(self, shape: &[usize]) -> Digest {
        match self {
            Self::TDigest => Digest::TDigest(TensorDigest::new(shape)),
            Self::QuantileSpine => Digest::QuantileSpine(TensorDigest::new(shape)),
        }
    }
}

impl Digest {
    pub fn update(&mut self, sample: &[f32]) {
        match self {
            Self::TDigest(digest) => digest.update(sample),
            Self::QuantileSpine(digest) => digest.update(sample),
        }
    }

    pub fn flush(&mut self) {
        match self {
            Self::TDigest(digest) => digest.flush(),
            Self::QuantileSpine(digest) => digest.flush(),
        }
    }

    pub fn quantile(&mut self, q: f32) -> Vec<f32> {
        match self {
            Self::TDigest(digest) => digest.quantile(q),
            Self::QuantileSpine(digest) => digest.quantile(q),
        }
    }

    pub fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        match self {
            Self::TDigest(digest) => digest.quantiles(qs),
            Self::QuantileSpine(digest) => digest.quantiles(qs),
        }
    }
}
