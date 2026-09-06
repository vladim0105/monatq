use rayon::prelude::*;
use std::borrow::Cow;
use std::mem;
#[cfg(feature = "visualize")]
use std::sync::atomic::AtomicBool;
use wide::f32x8;

use crate::{
    TensorValue,
    block::BlockLayout,
    distribution::Distribution,
    kernels::{DigestKernel, TDigest, TDigestConfig, sealed},
    tensor_digest::StorageOperations,
};

/// Flat-array TDigestStorage.
///
/// All centroid storage lives in contiguous arrays owned by this struct.
/// Element `e` occupies `centroids_*[e * max_centroids .. e * max_centroids + n_centroids[e]]`.
pub struct TDigestStorage<T: TensorValue> {
    layout: BlockLayout,
    compression: usize,

    // Row-major input buffer: sample s, element i → row_buffer[s * numel + i].
    row_buffer: Vec<T>,
    buffer_capacity: usize,
    n_buffered: usize,

    // Per-element centroid storage.
    max_centroids: usize,
    centroids_means: Vec<f32>,
    centroids_weights: Vec<u32>,
    n_centroids: Vec<usize>,
    total_weights: Vec<u32>,
    mins: Vec<T>,
    maxs: Vec<T>,
}

pub(crate) const TDIGEST_KERNEL_TAG: u8 = 0x54;

/// Snapshot format revision. Bump whenever the field layout below changes.
const TDIGEST_FORMAT_VERSION: u16 = 5;

/// Persistent summary only. Borrow on save; deserialize into owned arrays on load.
/// Ingestion workspace and capacities derived from compression are never serialized.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "T: serde::Serialize",
    deserialize = "T: serde::de::DeserializeOwned"
))]
struct TDigestSnapshot<'a, T: TensorValue> {
    kernel_tag: u8,
    format_version: u16,
    dtype_tag: u8,
    layout: Cow<'a, BlockLayout>,
    compression: usize,
    centroids_means: Cow<'a, [f32]>,
    centroids_weights: Cow<'a, [u32]>,
    n_centroids: Cow<'a, [usize]>,
    total_weights: Cow<'a, [u32]>,
    mins: Cow<'a, [T]>,
    maxs: Cow<'a, [T]>,
}

/// Leading fields shared by every TDigest snapshot.
#[derive(serde::Deserialize)]
struct TDigestHeader {
    kernel_tag: u8,
    format_version: u16,
    dtype_tag: u8,
}

/// Identify the element type of an uncompressed TDigest payload without decoding its state.
pub(crate) fn peek_dtype_tag(payload: &[u8]) -> Option<u8> {
    let header: TDigestHeader = bincode2::deserialize(payload).ok()?;
    (header.kernel_tag == TDIGEST_KERNEL_TAG).then_some(header.dtype_tag)
}

impl<T: TensorValue> TDigestStorage<T> {
    /// Create a new digest for tensors of the given `shape` (row-major).
    ///
    /// `compression` controls the T-Digest accuracy/memory trade-off: higher values
    /// keep more centroids and give more accurate quantile estimates. A value of 100
    /// is a reasonable default.
    pub fn new(shape: &[usize], compression: usize) -> Self {
        Self::with_layout(BlockLayout::default_for(shape), compression)
    }

    pub(crate) fn with_layout(layout: BlockLayout, compression: usize) -> Self {
        let input_numel = layout.input_numel();
        let buffer_capacity = compression * 2;
        let max_centroids = 6 * compression + 10;
        let state_count = layout.block_count();
        let row_len = if layout.is_elementwise() {
            input_numel * buffer_capacity
        } else {
            0
        };
        Self {
            layout,
            compression,
            row_buffer: vec![T::min_sentinel(); row_len],
            buffer_capacity,
            n_buffered: 0,
            max_centroids,
            centroids_means: vec![0.0; state_count * max_centroids],
            centroids_weights: vec![0; state_count * max_centroids],
            n_centroids: vec![0; state_count],
            total_weights: vec![0; state_count],
            mins: vec![T::min_sentinel(); state_count],
            maxs: vec![T::max_sentinel(); state_count],
        }
    }

    /// Total weight accumulated at atomic block `idx`.
    pub fn total_weight(&self, idx: usize) -> u32 {
        self.total_weights[idx]
    }

    /// Add one tensor sample. `data` must be row-major with `len == input_numel`.
    pub fn update(&mut self, data: &[T]) {
        let input_numel = self.layout.input_numel();
        assert_eq!(
            data.len(),
            input_numel,
            "data length {} does not match input element count {}",
            data.len(),
            input_numel
        );

        if !self.layout.is_elementwise() {
            self.process_data(data, 1);
            return;
        }
        let s = self.n_buffered;
        self.row_buffer[s * input_numel..(s + 1) * input_numel].copy_from_slice(data);
        self.n_buffered += 1;
        if self.n_buffered == self.buffer_capacity {
            self.flush();
        }
    }

    /// Flush all buffered samples into the per-element digests.
    pub fn flush(&mut self) {
        if self.n_buffered == 0 {
            return;
        }

        let n = self.n_buffered;
        let data_len = n * self.layout.input_numel();
        // Move the allocation out temporarily so processing can borrow it while mutating
        // the digest arrays, without cloning the full N1 input buffer.
        let row_buffer = mem::take(&mut self.row_buffer);
        self.process_data(&row_buffer[..data_len], n);
        self.row_buffer = row_buffer;
        self.n_buffered = 0;
    }

    fn process_data(&mut self, data: &[T], n: usize) {
        let input_numel = self.layout.input_numel();
        let max_centroids = self.max_centroids;
        let compression = self.compression;
        let layout = &self.layout;

        // Zip up the per-block mutable slices and process in parallel.
        let means_chunks = self.centroids_means.par_chunks_mut(max_centroids);
        let weights_chunks = self.centroids_weights.par_chunks_mut(max_centroids);
        let n_centroids = &mut self.n_centroids;
        let total_weights = &mut self.total_weights;
        let mins = &mut self.mins;
        let maxs = &mut self.maxs;

        means_chunks
            .zip(weights_chunks)
            .zip(n_centroids.par_iter_mut())
            .zip(total_weights.par_iter_mut())
            .zip(mins.par_iter_mut())
            .zip(maxs.par_iter_mut())
            .enumerate()
            .for_each_init(
                || Vec::with_capacity(n),
                |new_values: &mut Vec<T>,
                 (e, (((((e_means, e_weights), e_nc), e_tw), e_min), e_max))| {
                    // Reuse one scratch vector per worker to avoid per-element allocation churn.
                    new_values.clear();
                    for row in data.chunks_exact(input_numel) {
                        new_values.extend(layout.indices(e).map(|i| row[i]));
                    }
                    new_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    if let Some(&batch_min) = new_values.first() {
                        if batch_min < *e_min {
                            *e_min = batch_min;
                        }
                    }
                    if let Some(&batch_max) = new_values.last() {
                        if batch_max > *e_max {
                            *e_max = batch_max;
                        }
                    }

                    compress::<true, T>(
                        e_means,
                        e_weights,
                        e_nc,
                        e_tw,
                        new_values,
                        &[],
                        compression,
                        max_centroids,
                    );
                },
            );
    }

    /// Compute a single quantile at every position. Returns a flat row-major `Vec<f32>`.
    pub fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.flush();
        self.quantile_no_flush(q)
    }

    /// Compute multiple quantiles at every position.
    pub fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.flush();
        qs.iter().map(|&q| self.quantile_no_flush(q)).collect()
    }

    /// Classify the distribution at every position. Returns one `Distribution` per element.
    pub fn analyze(&mut self) -> Vec<Distribution> {
        self.flush();
        let max_centroids = self.max_centroids;
        self.centroids_means
            .par_chunks(max_centroids)
            .zip(self.centroids_weights.par_chunks(max_centroids))
            .zip(self.n_centroids.par_iter())
            .zip(self.total_weights.par_iter())
            .zip(self.mins.par_iter())
            .zip(self.maxs.par_iter())
            .map(|(((((means, weights), &nc), &tw), &min_v), &max_v)| {
                analyze_element(
                    &means[..nc],
                    &weights[..nc],
                    tw as f32,
                    min_v.to_f32(),
                    max_v.to_f32(),
                )
            })
            .collect()
    }

    /// Query multiple quantiles for one atomic block by flat block index.
    pub fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.flush();
        let start = idx * self.max_centroids;
        let nc = self.n_centroids[idx];
        let tw = self.total_weights[idx] as f32;
        let means = &self.centroids_means[start..start + nc];
        let weights = &self.centroids_weights[start..start + nc];
        let min_v = self.mins[idx].to_f32();
        let max_v = self.maxs[idx].to_f32();
        qs.iter()
            .map(|&q| quantile_from_centroids(means, weights, tw, min_v, max_v, q))
            .collect()
    }

    /// Merge the selected flat-indexed cells into a new one-element digest.
    pub fn merge_cells(&mut self, indices: &[usize]) -> Self {
        self.flush();

        let mut merged = TDigestStorage::new(&[1], self.compression);
        if indices.is_empty() {
            return merged;
        }

        // Collect all centroids from every source element into one flat list, then do a
        // single compression pass using the final combined total weight.  This matches the
        // C t-digest merge (combine → sort → one-pass compress) and avoids the centroid
        // overflow that occurs when merging element-by-element: early passes see a tiny
        // new_total, so the normalizer is large, the per-centroid budget is tight, and the
        // output count blows past max_centroids.
        let mut all: Vec<(f32, u32)> = Vec::new();
        for &idx in indices {
            let start = idx * self.max_centroids;
            let nc = self.n_centroids[idx];
            for i in 0..nc {
                all.push((
                    self.centroids_means[start + i],
                    self.centroids_weights[start + i],
                ));
            }
            update_min(&mut merged.mins[0], self.mins[idx]);
            update_max(&mut merged.maxs[0], self.maxs[idx]);
        }

        all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let (all_means, all_weights): (Vec<f32>, Vec<u32>) = all.into_iter().unzip();

        compress::<false, f32>(
            &mut merged.centroids_means[..merged.max_centroids],
            &mut merged.centroids_weights[..merged.max_centroids],
            &mut merged.n_centroids[0],
            &mut merged.total_weights[0],
            &all_means,
            &all_weights,
            merged.compression,
            merged.max_centroids,
        );

        merged
    }

    /// Merge the H×W spatial cells for each specified channel index into one digest.
    ///
    /// A "channel" is a contiguous block of `H×W` flat elements.  For a 4-D tensor
    /// `[B, C, H, W]` the channel flat index is `b * C + c`; for 3-D it is `c`.
    /// Each channel is compressed independently first, then the compressed centroid sets
    /// are combined — avoiding f32 precision loss when total weight exceeds ~2^26.
    pub fn merge_channels(&mut self, channel_indices: &[usize]) -> Self {
        let hw = self.spatial_size();

        // Phase 1: merge each channel's cells independently via merge_cells.
        // Phase 2: collect the compressed per-channel centroids and do one final compress.
        let mut all: Vec<(f32, u32)> = Vec::new();
        let mut min = T::min_sentinel();
        let mut max = T::max_sentinel();

        for &ch in channel_indices {
            let ch_digest = self.merge_cells(&(ch * hw..(ch + 1) * hw).collect::<Vec<_>>());
            let nc = ch_digest.n_centroids[0];
            for i in 0..nc {
                all.push((ch_digest.centroids_means[i], ch_digest.centroids_weights[i]));
            }
            update_min(&mut min, ch_digest.mins[0]);
            update_max(&mut max, ch_digest.maxs[0]);
        }

        let mut merged = TDigestStorage::new(&[1], self.compression);
        if all.is_empty() {
            return merged;
        }
        merged.mins[0] = min;
        merged.maxs[0] = max;
        all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let (all_means, all_weights): (Vec<f32>, Vec<u32>) = all.into_iter().unzip();

        compress::<false, f32>(
            &mut merged.centroids_means[..merged.max_centroids],
            &mut merged.centroids_weights[..merged.max_centroids],
            &mut merged.n_centroids[0],
            &mut merged.total_weights[0],
            &all_means,
            &all_weights,
            merged.compression,
            merged.max_centroids,
        );
        merged
    }

    /// Merge every atomic block exactly once into one digest.
    pub fn merge_all(&mut self) -> Self {
        self.flush();
        self.merge_cells(&(0..self.layout.block_count()).collect::<Vec<_>>())
    }

    /// Flush pending data and return a zstd-compressed bincode snapshot.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&mut self) -> std::io::Result<Vec<u8>>
    where
        T: serde::Serialize,
    {
        self.flush();
        let payload = bincode2::serialize(&TDigestSnapshot {
            kernel_tag: TDIGEST_KERNEL_TAG,
            format_version: TDIGEST_FORMAT_VERSION,
            dtype_tag: T::DTYPE_TAG,
            layout: Cow::Borrowed(&self.layout),
            compression: self.compression,
            centroids_means: Cow::Borrowed(&self.centroids_means),
            centroids_weights: Cow::Borrowed(&self.centroids_weights),
            n_centroids: Cow::Borrowed(&self.n_centroids),
            total_weights: Cow::Borrowed(&self.total_weights),
            mins: Cow::Borrowed(&self.mins),
            maxs: Cow::Borrowed(&self.maxs),
        })
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        zstd::encode_all(payload.as_slice(), 3).map_err(std::io::Error::other)
    }

    /// Load and decompress a snapshot from memory.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        let payload = zstd::decode_all(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Self::from_payload(&payload)
    }

    pub(crate) fn from_payload(payload: &[u8]) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        let header: TDigestHeader = bincode2::deserialize(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if header.kernel_tag != TDIGEST_KERNEL_TAG {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "not a TDigest snapshot: kernel tag {} but expected {TDIGEST_KERNEL_TAG}",
                    header.kernel_tag
                ),
            ));
        }
        if header.format_version != TDIGEST_FORMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported TDigest snapshot version {} but expected {TDIGEST_FORMAT_VERSION}",
                    header.format_version
                ),
            ));
        }
        if header.dtype_tag != T::DTYPE_TAG {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "dtype mismatch: snapshot contains tag {} but expected {}",
                    header.dtype_tag,
                    T::DTYPE_TAG
                ),
            ));
        }
        let loaded: TDigestSnapshot<T> = bincode2::deserialize(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        loaded
            .layout
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let invalid = |message| std::io::Error::new(std::io::ErrorKind::InvalidData, message);
        if loaded.compression == 0 {
            return Err(invalid("snapshot compression must be greater than zero"));
        }
        let expected_buffer_capacity = loaded
            .compression
            .checked_mul(2)
            .ok_or_else(|| invalid("snapshot compression overflows buffer capacity"))?;
        let expected_max_centroids = loaded
            .compression
            .checked_mul(6)
            .and_then(|value| value.checked_add(10))
            .ok_or_else(|| invalid("snapshot compression overflows centroid capacity"))?;
        let states = loaded.layout.block_count();
        let centroid_slots = states
            .checked_mul(expected_max_centroids)
            .ok_or_else(|| invalid("snapshot centroid array length overflows usize"))?;
        if loaded.n_centroids.len() != states
            || loaded.total_weights.len() != states
            || loaded.mins.len() != states
            || loaded.maxs.len() != states
            || loaded.centroids_means.len() != centroid_slots
            || loaded.centroids_weights.len() != centroid_slots
        {
            return Err(invalid("snapshot arrays do not match block layout"));
        }
        if loaded
            .n_centroids
            .iter()
            .any(|&count| count > expected_max_centroids)
        {
            return Err(invalid("snapshot centroid count exceeds capacity"));
        }

        let row_slots = if loaded.layout.is_elementwise() {
            loaded
                .layout
                .input_numel()
                .checked_mul(expected_buffer_capacity)
                .ok_or_else(|| invalid("snapshot row buffer length overflows usize"))?
        } else {
            0
        };
        Ok(Self {
            layout: loaded.layout.into_owned(),
            compression: loaded.compression,
            buffer_capacity: expected_buffer_capacity,
            max_centroids: expected_max_centroids,
            row_buffer: vec![T::min_sentinel(); row_slots],
            n_buffered: 0,
            centroids_means: loaded.centroids_means.into_owned(),
            centroids_weights: loaded.centroids_weights.into_owned(),
            n_centroids: loaded.n_centroids.into_owned(),
            total_weights: loaded.total_weights.into_owned(),
            mins: loaded.mins.into_owned(),
            maxs: loaded.maxs.into_owned(),
        })
    }

    /// Number of atomic blocks per channel in compact block geometry.
    fn spatial_size(&self) -> usize {
        let shape = self.layout.shape();
        let ndim = shape.len();
        if ndim < 2 {
            self.layout.block_count()
        } else {
            shape[ndim - 2] * shape[ndim - 1]
        }
    }

    fn quantile_no_flush(&self, q: f32) -> Vec<f32> {
        let max_centroids = self.max_centroids;
        self.centroids_means
            .par_chunks(max_centroids)
            .zip(self.centroids_weights.par_chunks(max_centroids))
            .zip(self.n_centroids.par_iter())
            .zip(self.total_weights.par_iter())
            .zip(self.mins.par_iter())
            .zip(self.maxs.par_iter())
            .map(|(((((means, weights), &nc), &tw), &min_v), &max_v)| {
                quantile_from_centroids(
                    &means[..nc],
                    &weights[..nc],
                    tw as f32,
                    min_v.to_f32(),
                    max_v.to_f32(),
                    q,
                )
            })
            .collect()
    }
}

impl<T: TensorValue> StorageOperations<T> for TDigestStorage<T> {
    fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    fn input_numel(&self) -> usize {
        self.layout.input_numel()
    }
    fn input_shape(&self) -> &[usize] {
        self.layout.input_shape()
    }
    fn block_count(&self) -> usize {
        self.layout.block_count()
    }
    fn block_axis(&self) -> usize {
        self.layout.axis()
    }
    fn blocks_per_axis(&self) -> usize {
        self.layout.blocks_per_axis()
    }
    fn block_size(&self) -> Option<usize> {
        self.layout.block_size()
    }
    fn total_weight(&self, idx: usize) -> crate::Result<u32> {
        crate::error::check_index(idx, self.layout.block_count())?;
        Ok(self.total_weight(idx))
    }
    fn update(&mut self, data: &[T]) -> crate::Result<()> {
        crate::error::check_sample_len(data.len(), self.layout.input_numel())?;
        self.update(data);
        Ok(())
    }
    fn flush(&mut self) {
        self.flush()
    }
    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.quantile(q)
    }
    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.quantiles(qs)
    }
    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> crate::Result<Vec<f32>> {
        crate::error::check_index(idx, self.layout.block_count())?;
        Ok(self.cell_quantiles(idx, qs))
    }
    fn merge_cells(&mut self, indices: &[usize]) -> crate::Result<Self> {
        for &idx in indices {
            crate::error::check_index(idx, self.layout.block_count())?;
        }
        Ok(TDigestStorage::merge_cells(self, indices))
    }
    fn merge_channels(&mut self, channel_indices: &[usize]) -> crate::Result<Self> {
        let hw = self.spatial_size();
        for &channel in channel_indices {
            let end = channel
                .checked_add(1)
                .and_then(|value| value.checked_mul(hw))
                .and_then(|value| value.checked_sub(1))
                .ok_or(crate::Error::IndexOutOfBounds {
                    index: usize::MAX,
                    numel: self.layout.block_count(),
                })?;
            crate::error::check_index(end, self.layout.block_count())?;
        }
        Ok(TDigestStorage::merge_channels(self, channel_indices))
    }
    fn merge_all(&mut self) -> crate::Result<Self> {
        Ok(TDigestStorage::merge_all(self))
    }
    fn analyze(&mut self) -> crate::Result<Vec<crate::Distribution>> {
        Ok(TDigestStorage::analyze(self))
    }
    fn without_zeros(&mut self) -> crate::Result<Self> {
        Ok(TDigestStorage::without_zeros(self))
    }
    fn to_bytes(&mut self) -> crate::Result<Vec<u8>>
    where
        T: serde::Serialize,
    {
        TDigestStorage::to_bytes(self).map_err(crate::Error::from_snapshot_io)
    }
    fn from_bytes(bytes: &[u8]) -> crate::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        TDigestStorage::from_bytes(bytes).map_err(crate::Error::from_snapshot_io)
    }
    fn from_payload(payload: &[u8]) -> crate::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        TDigestStorage::from_payload(payload).map_err(crate::Error::from_snapshot_io)
    }
    #[cfg(feature = "visualize")]
    fn visualize(&mut self) -> crate::Result<()> {
        TDigestStorage::visualize(self)
    }
    #[cfg(feature = "visualize")]
    fn visualize_until(&mut self, stop: &std::sync::atomic::AtomicBool) -> crate::Result<()> {
        TDigestStorage::visualize_until(self, stop)
    }
}

impl<T: TensorValue> DigestKernel<T> for TDigest {
    type Config = TDigestConfig;
}

impl<T: TensorValue> sealed::Kernel<T> for TDigest {
    type Storage = TDigestStorage<T>;

    fn create_storage(
        shape: &[usize],
        config: <TDigest as DigestKernel<T>>::Config,
    ) -> Self::Storage {
        TDigestStorage::new(shape, config.compression)
    }

    fn create_storage_with_layout(
        layout: BlockLayout,
        config: <TDigest as DigestKernel<T>>::Config,
    ) -> Self::Storage {
        TDigestStorage::with_layout(layout, config.compression)
    }
}

impl<T: TensorValue> TDigestStorage<T> {
    /// Launch a blocking HTTP visualizer server.
    /// Default port: 7777. Override with the `MONATQ_PORT` environment variable.
    #[cfg(feature = "visualize")]
    pub fn visualize(&mut self) -> crate::Result<()> {
        crate::server::serve(self)
    }

    /// Launch a blocking HTTP visualizer server that exits when `stop` is set.
    #[cfg(feature = "visualize")]
    pub fn visualize_until(&mut self, stop: &AtomicBool) -> crate::Result<()> {
        crate::server::serve_until(self, stop)
    }

    /// Return a copy with centroids centered at zero removed.
    ///
    /// This is intended for visualization of sparse tensors where exact zeros dominate the
    /// estimated density.
    pub fn without_zeros(&mut self) -> Self {
        self.flush();
        let mut filtered = TDigestStorage::with_layout(self.layout.clone(), self.compression);
        let eps = 1e-12_f32;

        for e in 0..self.layout.block_count() {
            let src_start = e * self.max_centroids;
            let dst_start = e * filtered.max_centroids;
            let nc = self.n_centroids[e];
            let src_means = &self.centroids_means[src_start..src_start + nc];
            let src_weights = &self.centroids_weights[src_start..src_start + nc];

            let mut out_nc = 0usize;
            let mut out_tw = 0u32;
            let mut out_min = f32::INFINITY;
            let mut out_max = f32::NEG_INFINITY;

            for i in 0..nc {
                if src_means[i].abs() <= eps {
                    continue;
                }
                filtered.centroids_means[dst_start + out_nc] = src_means[i];
                filtered.centroids_weights[dst_start + out_nc] = src_weights[i];
                out_nc += 1;
                out_tw += src_weights[i];
                out_min = out_min.min(src_means[i]);
                out_max = out_max.max(src_means[i]);
            }

            filtered.n_centroids[e] = out_nc;
            filtered.total_weights[e] = out_tw;
            if out_nc == 0 {
                filtered.mins[e] = T::from_f32(0.0);
                filtered.maxs[e] = T::from_f32(0.0);
                continue;
            }

            filtered.mins[e] = if self.mins[e].is_nonzero() {
                self.mins[e]
            } else {
                T::from_f32(out_min)
            };
            filtered.maxs[e] = if self.maxs[e].is_nonzero() {
                self.maxs[e]
            } else {
                T::from_f32(out_max)
            };
        }

        filtered
    }
}

fn analyze_element(
    means: &[f32],
    weights: &[u32],
    total_weight: f32,
    min_v: f32,
    max_v: f32,
) -> Distribution {
    if means.is_empty() {
        return Distribution::Unknown;
    }
    crate::distribution::classify(|q| {
        quantile_from_centroids(means, weights, total_weight, min_v, max_v, q)
    })
}

#[inline]
fn update_min<T: PartialOrd + Copy>(current: &mut T, candidate: T) {
    if candidate < *current {
        *current = candidate;
    }
}

#[inline]
fn update_max<T: PartialOrd + Copy>(current: &mut T, candidate: T) {
    if candidate > *current {
        *current = candidate;
    }
}

/// Merge sorted `incoming` into the centroid arrays for one element.
///
/// Merge `incoming` values into the per-element t-digest stored in `e_*`.
///
/// `UNIT` is a **compile-time** boolean that selects between two calling modes.
/// Using a const generic (rather than a runtime flag) lets the compiler monomorphize
/// two separate versions, eliminating all dead branches at compile time — which matters
/// because the `if UNIT` guards appear inside the hot Phase 3 loop.
///
/// - `UNIT = true` (flush path): every incoming value has weight 1.0.  `incoming_weights`
///   is ignored (pass `&[]`).  Phase 3 uses an 8-wide SIMD fast path.
/// - `UNIT = false` (merge path): incoming values carry arbitrary weights from
///   `incoming_weights`.  u64 is used for `cumulative`/`new_total` to stay correct
///   when the sum of two u32-bounded digests exceeds u32::MAX.
#[allow(clippy::too_many_arguments)]
fn compress<const UNIT: bool, T: TensorValue>(
    e_means: &mut [f32],
    e_weights: &mut [u32],
    e_nc: &mut usize,
    e_tw: &mut u32,
    incoming: &[T],
    incoming_weights: &[u32], // ignored when UNIT = true
    compression: usize,
    max_centroids: usize,
) {
    if UNIT {
        debug_assert!(
            incoming_weights.is_empty(),
            "UNIT=true compress called with non-empty incoming_weights"
        );
    }

    if incoming.is_empty() {
        return;
    }

    let old_nc = *e_nc;
    let incoming_total: u64 = if UNIT {
        incoming.len() as u64
    } else {
        incoming_weights.iter().map(|&w| w as u64).sum()
    };
    let new_total: u64 = *e_tw as u64 + incoming_total;
    let mut out_means: Vec<f32> = Vec::with_capacity(max_centroids);
    let mut out_weights: Vec<u32> = Vec::with_capacity(max_centroids);

    let mut cur_mean = 0.0f32;
    let mut cur_weight = 0u64;
    let mut cumulative = 0u64;
    let normalizer: f64 = if new_total > 1 {
        compression as f64
            / (2.0 * std::f64::consts::PI * new_total as f64 * (new_total as f64).ln())
    } else {
        0.0
    };

    macro_rules! absorb {
        ($m:expr, $w:expr) => {{
            let (m, w) = ($m as f32, $w as u64);
            let proposed_weight = cur_weight + w;
            let q0 = cumulative as f64 / new_total as f64;
            let q2 = (cumulative + proposed_weight) as f64 / new_total as f64;
            let z = proposed_weight as f64 * normalizer;
            let should_add = cur_weight > 0 && z <= q0 * (1.0 - q0) && z <= q2 * (1.0 - q2);
            if should_add {
                cur_mean += (w as f64 / proposed_weight as f64) as f32 * (m - cur_mean);
                cur_weight = proposed_weight;
            } else {
                if cur_weight > 0 {
                    out_means.push(cur_mean);
                    out_weights.push(cur_weight as u32);
                }
                cumulative += cur_weight;
                cur_mean = m;
                cur_weight = w;
            }
        }};
    }

    let mut ci = 0;
    let mut ni = 0;

    // Phase 1: interleaved merge
    while ci < old_nc && ni < incoming.len() {
        if e_means[ci] <= incoming[ni].to_f32() {
            absorb!(e_means[ci], e_weights[ci]);
            ci += 1;
        } else {
            absorb!(
                incoming[ni].to_f32(),
                if UNIT { 1u32 } else { incoming_weights[ni] }
            );
            ni += 1;
        }
    }

    // Phase 2: drain existing centroids
    while ci < old_nc {
        absorb!(e_means[ci], e_weights[ci]);
        ci += 1;
    }

    // Phase 3: drain incoming
    while ni < incoming.len() {
        if UNIT && ni + 8 <= incoming.len() {
            let proposed_weight = cur_weight + 8;
            let q0 = cumulative as f64 / new_total as f64;
            let q2 = (cumulative + proposed_weight) as f64 / new_total as f64;
            let z = proposed_weight as f64 * normalizer;
            let chunk: [f32; 8] = std::array::from_fn(|k| incoming[ni + k].to_f32());
            if cur_weight > 0 && z <= q0 * (1.0 - q0) && z <= q2 * (1.0 - q2) {
                let chunk_sum = f32x8::from(chunk).reduce_add();
                cur_mean += (chunk_sum - 8.0 * cur_mean) / proposed_weight as f32;
                cur_weight = proposed_weight;
                ni += 8;
                continue;
            }
        }
        absorb!(
            incoming[ni].to_f32(),
            if UNIT { 1u32 } else { incoming_weights[ni] }
        );
        ni += 1;
    }

    if cur_weight > 0 {
        out_means.push(cur_mean);
        out_weights.push(cur_weight as u32);
    }

    let result_nc = out_means.len();
    assert!(
        result_nc <= max_centroids,
        "compress: centroid count {result_nc} exceeds max_centroids {max_centroids}; \
         increase compression or max_centroids headroom"
    );

    e_means[..result_nc].copy_from_slice(&out_means);
    e_weights[..result_nc].copy_from_slice(&out_weights);
    *e_nc = result_nc;
    *e_tw = new_total as u32;
}

/// Standard TDigest quantile via linear scan + interpolation.
/// Uses a SIMD chunk-skip loop to locate the target centroid in O(nc/8) SIMD ops.
fn quantile_from_centroids(
    means: &[f32],
    weights: &[u32],
    total_weight: f32,
    min_v: f32,
    max_v: f32,
    q: f32,
) -> f32 {
    if total_weight <= 0.0 || means.is_empty() {
        return 0.0;
    }
    if q <= 0.0 {
        return min_v;
    }
    if q >= 1.0 {
        return max_v;
    }
    if means.len() == 1 {
        return means[0];
    }

    let target = q * total_weight;

    // Left tail: one sample is known to be at min, so the effective interpolation span is
    // w0/2 - 1 steps (not w0/2).  Guard against w0 <= 2 where the denominator collapses.
    let first_right = 1.0 + weights[0] as f32 / 2.0;
    if target <= first_right {
        if weights[0] <= 1 {
            return min_v;
        }
        let half_w0 = weights[0] as f32 / 2.0;
        let denom = if half_w0 > 1.0 {
            half_w0 - 1.0
        } else {
            half_w0
        };
        let t = ((target - 1.0) / denom).clamp(0.0, 1.0);
        return min_v + t * (means[0] - min_v);
    }

    // Right tail: one sample is known to be at max, so the effective span is w_last/2 - 1.
    let last = means.len() - 1;
    let last_left = total_weight - 1.0 - weights[last] as f32 / 2.0;
    if target >= last_left {
        if weights[last] <= 1 {
            return max_v;
        }
        let half_wl = weights[last] as f32 / 2.0;
        let denom = if half_wl > 1.0 {
            half_wl - 1.0
        } else {
            half_wl
        };
        let u = total_weight - target;
        let t = ((u - 1.0).max(0.0) / denom).clamp(0.0, 1.0);
        return max_v - t * (max_v - means[last]);
    }

    // Interior: interpolate between adjacent centroid centres with singleton contraction.
    // A singleton centroid occupies exactly its mean value; shrink the interpolation span
    // by 0.5 on each side that is a singleton, matching the RedisBloom t-digest-c behaviour.
    let mut cumulative = 0.0f32;
    for i in 0..means.len() - 1 {
        let left_center = cumulative + weights[i] as f32 / 2.0;
        cumulative += weights[i] as f32;
        let right_center = cumulative + weights[i + 1] as f32 / 2.0;

        if target > right_center {
            continue;
        }

        let left_singleton = weights[i] == 1;
        let right_singleton = weights[i + 1] == 1;

        if left_singleton && target - left_center < 0.5 {
            return means[i];
        }
        if right_singleton && right_center - target <= 0.5 {
            return means[i + 1];
        }

        let left_unit = if left_singleton { 0.5 } else { 0.0 };
        let right_unit = if right_singleton { 0.5 } else { 0.0 };
        let z1 = (target - left_center - left_unit).max(0.0);
        let z2 = (right_center - target - right_unit).max(0.0);
        let denom = z1 + z2;
        if denom <= 0.0 {
            return (means[i] + means[i + 1]) / 2.0;
        }
        return (means[i] * z2 + means[i + 1] * z1) / denom;
    }

    max_v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_mode_never_allocates_or_uses_tensor_row_buffer() {
        let layout = BlockLayout::new(&[4, 1024], crate::BlockConfig::new(4, 1)).unwrap();
        let mut td = TDigestStorage::<f32>::with_layout(layout, 100);
        assert!(td.row_buffer.is_empty());
        td.update(&vec![1.0; 4096]);
        assert!(td.row_buffer.is_empty());
        assert_eq!(td.n_buffered, 0);
    }

    fn current_snapshot() -> TDigestSnapshot<'static, f32> {
        let mut storage = TDigestStorage::<f32>::new(&[2], 10);
        storage.update(&[1.0, 2.0]);
        let bytes = storage.to_bytes().unwrap();
        let payload = zstd::decode_all(bytes.as_slice()).unwrap();
        bincode2::deserialize(&payload).unwrap()
    }

    fn assert_rejected(
        tamper: impl FnOnce(&mut TDigestSnapshot<'static, f32>),
        expected_fragment: &str,
    ) {
        let mut snapshot = current_snapshot();
        tamper(&mut snapshot);
        let payload = bincode2::serialize(&snapshot).unwrap();
        let Err(error) = TDigestStorage::<f32>::from_payload(&payload) else {
            panic!("tampered snapshot must be rejected");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            error.to_string().contains(expected_fragment),
            "error {error:?} does not mention {expected_fragment:?}"
        );
    }

    #[test]
    fn snapshot_rejects_wrong_version_and_invalid_metadata() {
        assert_rejected(
            |snapshot| snapshot.format_version += 1,
            "unsupported TDigest snapshot version",
        );
        assert_rejected(
            |snapshot| snapshot.kernel_tag ^= 0xff,
            "not a TDigest snapshot",
        );
        assert_rejected(|snapshot| snapshot.dtype_tag = 1, "dtype mismatch");
        assert_rejected(|snapshot| snapshot.compression = 0, "compression");
        assert_rejected(
            |snapshot| snapshot.compression = usize::MAX,
            "buffer capacity",
        );
        assert_rejected(
            |snapshot| snapshot.compression = usize::MAX / 6,
            "centroid capacity",
        );
        assert_rejected(
            |snapshot| snapshot.n_centroids.to_mut()[0] = snapshot.compression * 6 + 11,
            "centroid count",
        );
        assert_rejected(
            |snapshot| snapshot.compression = (usize::MAX - 10) / 6,
            "centroid array length overflows",
        );
        for array in 0..6 {
            assert_rejected(
                |snapshot| match array {
                    0 => {
                        snapshot.centroids_means.to_mut().pop();
                    }
                    1 => {
                        snapshot.centroids_weights.to_mut().pop();
                    }
                    2 => {
                        snapshot.n_centroids.to_mut().pop();
                    }
                    3 => {
                        snapshot.total_weights.to_mut().pop();
                    }
                    4 => {
                        snapshot.mins.to_mut().pop();
                    }
                    _ => {
                        snapshot.maxs.to_mut().pop();
                    }
                },
                "arrays do not match block layout",
            );
        }
    }

    #[test]
    fn snapshot_omits_stale_workspace_and_rebuilds_it_on_load() {
        let mut storage = TDigestStorage::<f32>::new(&[2], 10);
        for step in 0..27 {
            storage.update(&[step as f32, -(step as f32)]);
        }
        let bytes = storage.to_bytes().unwrap();
        assert_eq!(storage.n_buffered, 0);
        assert!(storage.row_buffer.iter().any(|value| value.is_finite()));

        // Saving the same summary must not depend on stale input values in the workspace.
        storage.row_buffer.fill(f32::NAN);
        assert_eq!(storage.to_bytes().unwrap(), bytes);
        let mut restored = TDigestStorage::<f32>::from_bytes(&bytes).unwrap();
        assert_eq!(restored.buffer_capacity, 20);
        assert_eq!(restored.max_centroids, 70);
        assert_eq!(restored.n_buffered, 0);
        assert_eq!(restored.row_buffer, vec![f32::INFINITY; 40]);
        assert_eq!(
            restored.quantiles(&[0.0, 0.5, 1.0]),
            storage.quantiles(&[0.0, 0.5, 1.0])
        );
    }

    #[test]
    fn blocked_snapshot_load_does_not_allocate_row_workspace() {
        let layout = BlockLayout::new(&[4], crate::BlockConfig::new(2, 0)).unwrap();
        let mut storage = TDigestStorage::<f32>::with_layout(layout, 10);
        storage.update(&[1.0, 2.0, 3.0, 4.0]);
        let bytes = storage.to_bytes().unwrap();
        let restored = TDigestStorage::<f32>::from_bytes(&bytes).unwrap();
        assert!(restored.row_buffer.is_empty());
        assert_eq!(restored.n_buffered, 0);
        assert_eq!(restored.layout, storage.layout);
    }

    #[test]
    fn basic_quantile() {
        let mut td = TDigestStorage::new(&[3], 100);

        for i in 0..1000usize {
            let x = i as f32;
            let sample = [
                (x * 0.006_283_185).sin(), // ≈ sin(i), oscillates ±1, median ≈ 0
                (x * 0.006_283_185).cos(), // ≈ cos(i), oscillates ±1, median ≈ 0
                x / 1000.0,                // ramps 0..1, median ≈ 0.5
            ];
            td.update(&sample);
        }

        let q50 = td.quantile(0.5);
        assert_eq!(q50.len(), 3);

        // sin median ≈ 0.0
        assert!(q50[0].abs() < 0.1, "sin median {:.4} not near 0", q50[0]);
        // cos median ≈ 0.0
        assert!(q50[1].abs() < 0.1, "cos median {:.4} not near 0", q50[1]);
        // ramp median ≈ 0.5
        assert!(
            (q50[2] - 0.5).abs() < 0.05,
            "ramp median {:.4} not near 0.5",
            q50[2]
        );
    }
}
