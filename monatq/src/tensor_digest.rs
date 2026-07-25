use std::marker::PhantomData;

use crate::{
    TensorValue,
    kernels::{self, DigestKernel, QuantileSpine, RankKnot},
};

/// Operations shared by every kernel-specific storage layout.
///
/// This trait is crate-private so storage remains an implementation detail while the
/// public container can provide one statically dispatched implementation of its common API.
pub(crate) trait StorageOperations<T: TensorValue>: Sized {
    fn numel(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn total_weight(&self, idx: usize) -> u32;
    fn update(&mut self, data: &[T]);
    fn flush(&mut self);
    fn quantile(&mut self, q: f32) -> Vec<f32>;
    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>>;
    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32>;
    fn merge_cells(&mut self, indices: &[usize]) -> Self;
    fn merge_channels(&mut self, channel_indices: &[usize]) -> Self;
    fn merge_all(&mut self) -> Self;
    fn analyze(&mut self) -> Vec<crate::Distribution>;
    fn without_zeros(&mut self) -> Self;
    fn to_bytes(&mut self) -> std::io::Result<Vec<u8>>
    where
        T: serde::Serialize;
    fn from_bytes(bytes: &[u8]) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned;
    fn from_payload(payload: &[u8]) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned;
    #[cfg(feature = "visualize")]
    fn visualize(&mut self) -> std::io::Result<()>;
    #[cfg(feature = "visualize")]
    fn visualize_until(&mut self, stop: &std::sync::atomic::AtomicBool) -> std::io::Result<()>;
}

/// A tensor-wide approximate quantile digest using the statically selected kernel `K`.
///
/// The marker parameter selects a concrete optimized storage layout at compile time; no
/// runtime enum or dynamic dispatch is used.
#[repr(transparent)]
pub struct TensorDigest<T: TensorValue, K: DigestKernel<T>> {
    storage: <K as kernels::sealed::Kernel<T>>::Storage,
    marker: PhantomData<(T, K)>,
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

    pub(crate) fn from_storage(storage: <K as kernels::sealed::Kernel<T>>::Storage) -> Self {
        Self {
            storage,
            marker: PhantomData,
        }
    }

    /// Total number of elements (the product of the shape dimensions).
    pub fn numel(&self) -> usize {
        self.storage.numel()
    }

    /// Shape of the tensors tracked by this digest.
    pub fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    /// Total flushed sample weight at one flat-indexed element.
    pub fn total_weight(&self, idx: usize) -> u32 {
        self.storage.total_weight(idx)
    }

    /// Add one row-major tensor sample.
    pub fn update(&mut self, data: &[T]) {
        self.storage.update(data)
    }

    /// Flush buffered samples into the kernel storage.
    pub fn flush(&mut self) {
        self.storage.flush()
    }

    /// Compute one quantile at every tensor position.
    pub fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.storage.quantile(q)
    }

    /// Compute several quantiles at every tensor position.
    pub fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.storage.quantiles(qs)
    }

    /// Compute several quantiles for one flat-indexed tensor position.
    pub fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.storage.cell_quantiles(idx, qs)
    }

    /// Merge selected flat-indexed tensor positions into a one-position digest.
    ///
    /// Panics if the selected kernel does not yet implement merging.
    pub fn merge_cells(&mut self, indices: &[usize]) -> Self {
        Self::from_storage(self.storage.merge_cells(indices))
    }

    /// Merge selected leading-dimension channels into one channel digest.
    ///
    /// Panics if the selected kernel does not yet implement merging.
    pub fn merge_channels(&mut self, channel_indices: &[usize]) -> Self {
        Self::from_storage(self.storage.merge_channels(channel_indices))
    }

    /// Merge every tensor position into a one-position digest.
    ///
    /// Panics if the selected kernel does not yet implement merging.
    pub fn merge_all(&mut self) -> Self {
        Self::from_storage(self.storage.merge_all())
    }

    /// Analyze the distribution at every tensor position.
    pub fn analyze(&mut self) -> Vec<crate::Distribution> {
        self.storage.analyze()
    }

    /// Return a copy with values centered at zero removed.
    pub fn without_zeros(&mut self) -> Self {
        Self::from_storage(self.storage.without_zeros())
    }

    /// Serialize this digest.
    pub fn to_bytes(&mut self) -> std::io::Result<Vec<u8>>
    where
        T: serde::Serialize,
    {
        self.storage.to_bytes()
    }

    /// Save this digest to a file.
    pub fn save(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()>
    where
        T: serde::Serialize,
    {
        std::fs::write(path, self.to_bytes()?)
    }

    /// Deserialize a digest.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        <K::Storage as StorageOperations<T>>::from_bytes(bytes).map(Self::from_storage)
    }

    /// Load a digest from a file.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub(crate) fn from_payload(payload: &[u8]) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        <K::Storage as StorageOperations<T>>::from_payload(payload).map(Self::from_storage)
    }

    #[cfg(feature = "visualize")]
    pub fn visualize(&mut self) -> std::io::Result<()> {
        self.storage.visualize()
    }

    #[cfg(feature = "visualize")]
    pub fn visualize_until(&mut self, stop: &std::sync::atomic::AtomicBool) -> std::io::Result<()> {
        self.storage.visualize_until(stop)
    }
}

impl TensorDigest<f32, RankKnot> {
    /// Number of accepted samples, including samples flushed for this query.
    pub fn sample_count(&mut self) -> u64 {
        self.flush();
        self.storage.sample_count()
    }

    pub fn config(&self) -> &crate::RankKnotConfig {
        self.storage.config()
    }

    pub fn min(&mut self) -> Vec<f32> {
        self.storage.min()
    }

    pub fn max(&mut self) -> Vec<f32> {
        self.storage.max()
    }

    pub fn cell_min(&mut self, idx: usize) -> f32 {
        self.storage.cell_min(idx)
    }

    pub fn cell_max(&mut self, idx: usize) -> f32 {
        self.storage.cell_max(idx)
    }
}

impl<T: TensorValue> TensorDigest<T, QuantileSpine> {
    pub fn sample_count(&self) -> u32 {
        self.storage.sample_count()
    }

    pub fn config(&self) -> &crate::QuantileSpineConfig {
        self.storage.config()
    }

    pub fn link(&mut self) -> crate::SpineLink {
        self.storage.link()
    }

    pub fn min(&mut self) -> Vec<f32> {
        self.storage.min()
    }

    pub fn max(&mut self) -> Vec<f32> {
        self.storage.max()
    }

    pub fn cell_min(&mut self, idx: usize) -> f32 {
        self.storage.cell_min(idx)
    }

    pub fn cell_max(&mut self, idx: usize) -> f32 {
        self.storage.cell_max(idx)
    }

    pub fn recent_min(&mut self) -> Vec<f32> {
        self.storage.recent_min()
    }

    pub fn recent_max(&mut self) -> Vec<f32> {
        self.storage.recent_max()
    }

    pub fn zero_count(&mut self, idx: usize) -> u32 {
        self.storage.zero_count(idx)
    }

    pub fn secondary_atom(&mut self, idx: usize) -> Option<(f32, u32)> {
        self.storage.secondary_atom(idx)
    }

    pub fn surprise(&mut self, idx: usize) -> f32 {
        self.storage.surprise(idx)
    }

    pub fn regime(&mut self, idx: usize) -> crate::SpineRegime {
        self.storage.regime(idx)
    }
}
