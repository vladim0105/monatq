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
    /// New values each block collects before parallel compression. The digest buffers
    /// `ceil(buffer_capacity / block_len)` whole tensor rows, so with one-element blocks this
    /// is the number of buffered samples. When one sample already fills a block, or when this
    /// is zero, every sample is compressed immediately without a buffer.
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
    /// New values each block collects before compression, with the same meaning as
    /// [`RankKnotConfig::buffer_capacity`]. `None` uses `2 * compression`.
    pub buffer_capacity: Option<usize>,
}

impl TDigestConfig {
    pub(crate) fn effective_buffer_capacity(&self) -> usize {
        self.buffer_capacity
            .unwrap_or_else(|| self.compression.saturating_mul(2))
    }
}

impl Default for TDigestConfig {
    fn default() -> Self {
        Self {
            compression: 100,
            buffer_capacity: None,
        }
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
