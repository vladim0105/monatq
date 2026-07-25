pub(crate) mod quantile_spine;
pub(crate) mod tdigest;

use crate::{TensorValue, tensor_digest::StorageOperations};

/// Marker selecting the T-Digest kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct TDigest;

/// Marker selecting the Quantile Spine kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantileSpine;

/// Configuration for the T-Digest kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TDigestConfig {
    /// Accuracy/memory trade-off. Higher values retain more centroids.
    pub compression: usize,
}

impl Default for TDigestConfig {
    fn default() -> Self {
        Self { compression: 100 }
    }
}

/// A statically selected quantile kernel supported by [`crate::TensorDigest`].
///
/// This trait is sealed: downstream crates can name it for generic bounds but cannot
/// implement additional kernels. Kernel storage is deliberately absent from this public API.
#[allow(private_bounds)]
pub trait DigestKernel<T: TensorValue>: sealed::Kernel<T> {
    /// Public configuration accepted by [`crate::TensorDigest::with_config`].
    type Config: Default;
}

pub(crate) mod sealed {
    use super::*;

    pub(crate) trait Kernel<T: TensorValue>: Sized {
        type Storage: StorageOperations<T>;

        fn create_storage(
            shape: &[usize],
            config: <Self as DigestKernel<T>>::Config,
        ) -> Self::Storage
        where
            Self: DigestKernel<T>;
    }
}
