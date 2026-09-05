pub(crate) mod rankknot;
pub(crate) mod tdigest;

use crate::{
    BlockConfig, Result, TensorValue, block::BlockLayout, tensor_digest::StorageOperations,
};

/// Marker selecting the T-Digest kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct TDigest;

/// Marker selecting the RankKnot kernel.
///
/// Supports `f32` and `i32`. Summary state is `f32` for both, so an `i32` stream is
/// summarised at `f32` resolution and magnitudes above 2^24 round to the nearest
/// representable neighbour. The t-digest kernel has the same ceiling for `i32`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RankKnot;

/// Configuration for the RankKnot kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankKnotConfig {
    /// Number of complete tensor samples buffered before parallel compression.
    /// Zero bypasses buffering and updates immediately on every sample.
    pub buffer_capacity: usize,
}

impl Default for RankKnotConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 256,
        }
    }
}

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

        fn create_block_storage(
            shape: &[usize],
            config: <Self as DigestKernel<T>>::Config,
            blocks: BlockConfig,
        ) -> Result<Self::Storage>
        where
            Self: DigestKernel<T>,
        {
            let layout = BlockLayout::new(shape, blocks)?;
            Ok(Self::create_storage_with_layout(layout, config))
        }

        fn create_storage_with_layout(
            layout: BlockLayout,
            config: <Self as DigestKernel<T>>::Config,
        ) -> Self::Storage
        where
            Self: DigestKernel<T>;
    }
}
