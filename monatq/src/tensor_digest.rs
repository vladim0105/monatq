use std::marker::PhantomData;

use crate::{
    BlockConfig, Result, TensorValue,
    kernels::{self, DigestKernel, RankKnot},
};

/// Operations shared by every kernel-specific storage layout.
///
/// This trait is crate-private so storage remains an implementation detail while the
/// public container can provide one statically dispatched implementation of its common API.
pub(crate) trait StorageOperations<T: TensorValue>: Sized {
    fn shape(&self) -> &[usize];
    fn numel(&self) -> usize;
    fn block_shape(&self) -> &[usize];
    fn block_count(&self) -> usize;
    fn block_config(&self) -> BlockConfig;
    fn reshape(&mut self, shape: &[usize]) -> Result<()>;
    fn remap(&mut self, shape: &[usize], blocks: BlockConfig) -> Result<()>;
    fn permute(&mut self, axes: &[isize]) -> Result<()>;
    fn total_weight(&self, idx: usize) -> Result<u32>;
    fn update(&mut self, data: &[T]) -> Result<()>;
    fn flush(&mut self);
    fn quantile(&mut self, q: f32) -> Vec<f32>;
    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>>;
    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Result<Vec<f32>>;
    fn merge_cells(&mut self, indices: &[usize]) -> Result<Self>;
    fn merge_channels(&mut self, channel_indices: &[usize]) -> Result<Self>;
    fn merge_all(&mut self) -> Result<Self>;
    fn analyze(&mut self) -> Result<Vec<crate::Distribution>>;
    fn without_zeros(&mut self) -> Result<Self>;
    fn to_bytes(&mut self) -> Result<Vec<u8>>
    where
        T: serde::Serialize;
    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        T: serde::de::DeserializeOwned;
    fn from_payload(payload: &[u8]) -> Result<Self>
    where
        T: serde::de::DeserializeOwned;
    #[cfg(feature = "visualize")]
    fn visualize(&mut self) -> Result<()>;
    #[cfg(feature = "visualize")]
    fn visualize_until(&mut self, stop: &std::sync::atomic::AtomicBool) -> Result<()>;
}

/// A tensor-wide approximate quantile digest using the statically selected kernel `K`.
///
/// The marker parameter selects a concrete optimized storage layout at compile time; no
/// runtime enum or dynamic dispatch is used.
///
/// `K` defaults to [`RankKnot`], which implements the full contract in 208 bytes of state
/// per tensor position. Name a kernel explicitly to override it:
///
/// ```
/// use monatq::{TDigest, TensorDigest};
///
/// let default_kernel = TensorDigest::<f32>::new(&[2, 2]);
/// let t_digest = TensorDigest::<f32, TDigest>::new(&[2, 2]);
/// ```
#[repr(transparent)]
pub struct TensorDigest<T: TensorValue, K: DigestKernel<T> = RankKnot> {
    storage: <K as kernels::sealed::Kernel<T>>::Storage,
    marker: PhantomData<(T, K)>,
}

/// Reports the tensor geometry only.
///
/// Kernel storage is deliberately opaque, but a `Debug` impl is still worth having: without
/// one, `Result<TensorDigest, _>` cannot be used with `unwrap_err`, `expect_err`, or
/// `assert_eq!`, which makes fallible APIs awkward to test.
impl<T: TensorValue, K: DigestKernel<T>> std::fmt::Debug for TensorDigest<T, K> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorDigest")
            .field("shape", &self.shape())
            .field("block_shape", &self.block_shape())
            .finish_non_exhaustive()
    }
}

impl<T: TensorValue, K: DigestKernel<T>> TensorDigest<T, K> {
    /// Construct a tensor digest using the selected kernel's default configuration.
    pub fn new(shape: &[usize]) -> Self {
        Self::with_config(shape, K::Config::default())
    }

    /// Construct a tensor digest using an explicit kernel configuration.
    pub fn with_config(shape: &[usize], config: K::Config) -> Self {
        Self::from_storage(K::create_storage(shape, config))
    }

    /// Construct a digest with one-dimensional axis-local groups, selected by size or count.
    /// The axis, the positive size or count, and the resulting layout are validated.
    pub fn with_block_config(
        shape: &[usize],
        config: K::Config,
        blocks: BlockConfig,
    ) -> Result<Self> {
        K::create_block_storage(shape, config, blocks).map(Self::from_storage)
    }

    /// Construct a blocked digest with the kernel's default configuration.
    pub fn with_blocks(shape: &[usize], blocks: BlockConfig) -> Result<Self> {
        Self::with_block_config(shape, K::Config::default(), blocks)
    }

    pub(crate) fn from_storage(storage: <K as kernels::sealed::Kernel<T>>::Storage) -> Self {
        Self {
            storage,
            marker: PhantomData,
        }
    }

    /// Shape of the tensors accepted by [`Self::update`], as passed at construction.
    ///
    /// For a blocked digest, query results follow [`Self::block_shape`] instead.
    pub fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    /// Total number of elements (the product of the shape dimensions).
    pub fn numel(&self) -> usize {
        self.storage.numel()
    }

    /// Compact row-major atomic-block shape used by queries, selections, and merges.
    ///
    /// Equal to [`Self::shape`] for an elementwise digest.
    pub fn block_shape(&self) -> &[usize] {
        self.storage.block_shape()
    }

    /// Number of independently tracked statistical blocks (the product of
    /// [`Self::block_shape`]).
    pub fn block_count(&self) -> usize {
        self.storage.block_count()
    }

    /// The block grouping in resolved form: the axis is a nonnegative index, and a count is
    /// the effective count after clamping to the axis length. A size is reported as
    /// requested; the effective number of groups along the axis is `block_shape()[axis]`.
    pub fn block_config(&self) -> BlockConfig {
        self.storage.block_config()
    }

    /// Change the tensor shape accepted by [`Self::update`], keeping the current block grouping.
    ///
    /// Each block's full summary stays where it is, and observations still waiting in the
    /// input buffer stay with the elements they already describe. Nothing is flushed or
    /// recompressed. On failure the digest is left unchanged.
    ///
    /// `shape` must have the same number of elements. An elementwise digest accepts any such
    /// shape. A blocked digest is updated only when every pooled block still covers exactly
    /// the same elements; a layout that would split a pooled block is rejected because those
    /// observations can no longer be separated. See [`Self::remap`] to change the grouping as
    /// well.
    ///
    /// ```
    /// use monatq::TensorDigest;
    ///
    /// let mut digest = TensorDigest::<f32>::new(&[2, 3]);
    /// digest.update(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// digest.reshape(&[3, 2]).unwrap();
    /// assert_eq!(digest.shape(), &[3, 2]);
    /// assert_eq!(digest.quantile(1.0), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        self.storage.reshape(shape)
    }

    /// Change the tensor shape and the block grouping together.
    ///
    /// Preservation rules match [`Self::reshape`]: full per-block state and pending
    /// observations both survive, and a rejected call does not modify the digest.
    /// Elementwise digests accept any elementwise target with the same element count.
    /// A target that would split a pooled block, or combine blocks that are still tracked
    /// separately, is rejected with [`crate::Error::IncompatibleLayout`].
    pub fn remap(&mut self, shape: &[usize], blocks: BlockConfig) -> Result<()> {
        self.storage.remap(shape, blocks)
    }

    /// Reorder the tensor axes, like numpy's `transpose(axes)` or torch's `permute(*dims)`.
    ///
    /// Axis `i` of the new shape is axis `axes[i]` of the current one, so [`Self::update`]
    /// afterwards takes tensors in the permuted layout. `axes` must name every axis exactly
    /// once; negative axes count from the end. Unlike [`Self::reshape`], this moves elements,
    /// so every element's history travels with it: the digest at new index `(j0, j1, ...)` is
    /// the one that was at the old index whose coordinate along axis `axes[i]` is `ji`.
    ///
    /// A block grouping follows its axis, so blocked digests are never rejected: each block
    /// keeps covering the same elements, and [`Self::block_shape`] is permuted the same way
    /// as the tensor shape. Observations still waiting in the input buffer are reordered with
    /// the elements they describe. Nothing is flushed or recompressed. Invalid `axes` fail
    /// with [`crate::Error::InvalidConfig`] and leave the digest unchanged.
    ///
    /// ```
    /// use monatq::TensorDigest;
    ///
    /// let mut digest = TensorDigest::<f32>::new(&[2, 3]);
    /// digest.update(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// digest.permute(&[1, 0]).unwrap();
    /// assert_eq!(digest.shape(), &[3, 2]);
    /// assert_eq!(digest.quantile(1.0), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn permute(&mut self, axes: &[isize]) -> Result<()> {
        self.storage.permute(axes)
    }

    /// Total flushed observation weight for an atomic block.
    ///
    /// Fails with [`crate::Error::IndexOutOfBounds`] if `idx` is not a valid block index.
    pub fn total_weight(&self, idx: usize) -> Result<u32> {
        self.storage.total_weight(idx)
    }

    /// Add one row-major tensor sample.
    ///
    /// Fails with [`crate::Error::ShapeMismatch`] if `data` does not have exactly
    /// [`Self::numel`] elements. The digest is left untouched in that case.
    ///
    /// NaN input is a documented precondition rather than a checked one: it is not rejected
    /// here and will panic during compression (during update in direct/block mode, otherwise
    /// during a later flush).
    pub fn update(&mut self, data: &[T]) -> Result<()> {
        self.storage.update(data)
    }

    /// Flush buffered samples into the kernel storage.
    pub fn flush(&mut self) {
        self.storage.flush()
    }

    /// Compute one quantile per statistical block, in compact row-major block shape.
    pub fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.storage.quantile(q)
    }

    /// Compute several quantiles per statistical block.
    pub fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.storage.quantiles(qs)
    }

    /// Compute several quantiles for one atomic block index.
    ///
    /// Fails with [`crate::Error::IndexOutOfBounds`] if `idx` is not a valid block index.
    pub fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Result<Vec<f32>> {
        self.storage.cell_quantiles(idx, qs)
    }

    /// Merge selected atomic blocks into a scalar digest.
    ///
    /// Fails with [`crate::Error::IndexOutOfBounds`] for an invalid block index.
    pub fn merge_cells(&mut self, indices: &[usize]) -> Result<Self> {
        self.storage.merge_cells(indices).map(Self::from_storage)
    }

    /// Merge selected leading-dimension channels into one channel digest.
    ///
    /// Fails with [`crate::Error::Unsupported`] if the selected kernel does not implement
    /// merging, or [`crate::Error::IndexOutOfBounds`] for an invalid channel.
    pub fn merge_channels(&mut self, channel_indices: &[usize]) -> Result<Self> {
        self.storage
            .merge_channels(channel_indices)
            .map(Self::from_storage)
    }

    /// Merge every atomic block exactly once into a scalar digest.
    pub fn merge_all(&mut self) -> Result<Self> {
        self.storage.merge_all().map(Self::from_storage)
    }

    /// Analyze the distribution of every statistical block, in compact row-major order.
    ///
    /// Fails with [`crate::Error::Unsupported`] if the selected kernel does not implement
    /// analysis.
    pub fn analyze(&mut self) -> Result<Vec<crate::Distribution>> {
        self.storage.analyze()
    }

    /// Return a copy with values centered at zero removed.
    ///
    /// Fails with [`crate::Error::Unsupported`] if the selected kernel does not implement
    /// zero filtering.
    pub fn without_zeros(&mut self) -> Result<Self> {
        self.storage.without_zeros().map(Self::from_storage)
    }

    /// Serialize this digest.
    pub fn to_bytes(&mut self) -> Result<Vec<u8>>
    where
        T: serde::Serialize,
    {
        self.storage.to_bytes()
    }

    /// Save this digest to a file.
    pub fn save(&mut self, path: impl AsRef<std::path::Path>) -> Result<()>
    where
        T: serde::Serialize,
    {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(crate::Error::Io)
    }

    /// Deserialize a digest.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        <K::Storage as StorageOperations<T>>::from_bytes(bytes).map(Self::from_storage)
    }

    /// Load a digest from a file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        let bytes = std::fs::read(path).map_err(crate::Error::Io)?;
        Self::from_bytes(&bytes)
    }

    pub(crate) fn from_payload(payload: &[u8]) -> Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        <K::Storage as StorageOperations<T>>::from_payload(payload).map(Self::from_storage)
    }

    #[cfg(feature = "visualize")]
    pub fn visualize(&mut self) -> Result<()> {
        self.storage.visualize()
    }

    #[cfg(feature = "visualize")]
    pub fn visualize_until(&mut self, stop: &std::sync::atomic::AtomicBool) -> Result<()> {
        self.storage.visualize_until(stop)
    }
}

impl<T: TensorValue> TensorDigest<T, RankKnot> {
    /// Number of accepted samples, including samples flushed for this query.
    pub fn sample_count(&mut self) -> u64 {
        self.flush();
        self.storage.sample_count()
    }

    pub fn config(&self) -> &crate::RankKnotConfig {
        self.storage.config()
    }

    /// Minimum for each statistical block, in compact row-major order.
    pub fn min(&mut self) -> Vec<f32> {
        self.storage.min()
    }

    /// Maximum for each statistical block, in compact row-major order.
    pub fn max(&mut self) -> Vec<f32> {
        self.storage.max()
    }

    /// Fails with [`crate::Error::IndexOutOfBounds`] if `idx` is not a valid block index.
    pub fn cell_min(&mut self, idx: usize) -> Result<f32> {
        crate::error::check_index(idx, self.block_count())?;
        Ok(self.storage.cell_min(idx))
    }

    /// Fails with [`crate::Error::IndexOutOfBounds`] if `idx` is not a valid block index.
    pub fn cell_max(&mut self, idx: usize) -> Result<f32> {
        crate::error::check_index(idx, self.block_count())?;
        Ok(self.storage.cell_max(idx))
    }
}
