//! Shared backend adapter for repository benchmarks and accuracy reports.

use crate::{RankKnot, TDigest, TensorDigest};

/// Digest implementations exercised by backend comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    TDigest,
    RankKnot,
}

/// The single backend list used by performance and accuracy reporting.
pub const BACKENDS: &[Backend] = &[Backend::TDigest, Backend::RankKnot];

/// Type-erased adapter used only by repository tooling.
pub enum Digest {
    TDigest(TensorDigest<f32, TDigest>),
    RankKnot(TensorDigest<f32, RankKnot>),
}

impl Backend {
    pub fn create(self, shape: &[usize]) -> Digest {
        match self {
            Self::TDigest => Digest::TDigest(TensorDigest::new(shape)),
            Self::RankKnot => Digest::RankKnot(TensorDigest::new(shape)),
        }
    }
}

impl Digest {
    pub fn update(&mut self, sample: &[f32]) -> crate::Result<()> {
        match self {
            Self::TDigest(digest) => digest.update(sample),
            Self::RankKnot(digest) => digest.update(sample),
        }
    }

    pub fn flush(&mut self) {
        match self {
            Self::TDigest(digest) => digest.flush(),
            Self::RankKnot(digest) => digest.flush(),
        }
    }

    pub fn quantile(&mut self, q: f32) -> Vec<f32> {
        match self {
            Self::TDigest(digest) => digest.quantile(q),
            Self::RankKnot(digest) => digest.quantile(q),
        }
    }

    /// Merge every tensor position into a one-position digest.
    ///
    /// Returns [`crate::Error::Unsupported`] for kernels without a merge implementation, so
    /// reporting tools can render a placeholder instead of calling and crashing.
    pub fn merge_all(&mut self) -> crate::Result<Digest> {
        Ok(match self {
            Self::TDigest(digest) => Digest::TDigest(digest.merge_all()?),
            Self::RankKnot(digest) => Digest::RankKnot(digest.merge_all()?),
        })
    }

    pub fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        match self {
            Self::TDigest(digest) => digest.quantiles(qs),
            Self::RankKnot(digest) => digest.quantiles(qs),
        }
    }
}
