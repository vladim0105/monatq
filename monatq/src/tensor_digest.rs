use rayon::prelude::*;
#[cfg(feature = "visualize")]
use std::sync::atomic::AtomicBool;
use strum::IntoEnumIterator;
use wide::f32x8;

use crate::{
    TensorValue,
    distribution::{Distribution, N_PADDED, probe_points, ref_profiles},
};

/// Flat-array TensorDigest.
///
/// All centroid storage lives in contiguous arrays owned by this struct.
/// Element `e` occupies `centroids_*[e * max_centroids .. e * max_centroids + n_centroids[e]]`.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "T: serde::Serialize",
    deserialize = "T: serde::de::DeserializeOwned"
))]
pub struct TensorDigest<T: TensorValue> {
    dtype_tag: u8,
    shape: Vec<usize>,
    numel: usize,
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

impl<T: TensorValue> TensorDigest<T> {
    /// Create a new digest for tensors of the given `shape` (row-major).
    ///
    /// `compression` controls the T-Digest accuracy/memory trade-off: higher values
    /// keep more centroids and give more accurate quantile estimates. A value of 100
    /// is a reasonable default.
    pub fn new(shape: &[usize], compression: usize) -> Self {
        let numel = shape.iter().product::<usize>();
        let buffer_capacity = compression * 2;
        // Same as in t-digest-c by RedisBloom
        let max_centroids = 6 * compression + 10;

        Self {
            dtype_tag: T::DTYPE_TAG,
            shape: shape.to_vec(),
            numel,
            compression,
            row_buffer: vec![T::min_sentinel(); numel * buffer_capacity],
            buffer_capacity,
            n_buffered: 0,
            max_centroids,
            centroids_means: vec![0.0f32; numel * max_centroids],
            centroids_weights: vec![0u32; numel * max_centroids],
            n_centroids: vec![0usize; numel],
            total_weights: vec![0u32; numel],
            mins: vec![T::min_sentinel(); numel],
            maxs: vec![T::max_sentinel(); numel],
        }
    }

    /// Total number of elements (product of all shape dimensions).
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Shape of the tensors being tracked (as passed to `new`).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total weight (sample count) accumulated at element `idx`.
    pub fn total_weight(&self, idx: usize) -> u32 {
        self.total_weights[idx]
    }

    /// Add one tensor sample. `data` must be row-major with `len == numel()`.
    pub fn update(&mut self, data: &[T]) {
        assert_eq!(
            data.len(),
            self.numel,
            "data length {} does not match numel {}",
            data.len(),
            self.numel
        );

        let s = self.n_buffered;
        self.row_buffer[s * self.numel..(s + 1) * self.numel].copy_from_slice(data);
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
        let numel = self.numel;
        let max_centroids = self.max_centroids;
        let compression = self.compression;
        let row_buffer = &self.row_buffer;

        // Zip up the per-element mutable slices and process in parallel.
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
                    new_values.extend((0..n).map(|s| row_buffer[s * numel + e]));
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

        self.n_buffered = 0;
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

    /// Query multiple quantiles for a single element by flat index.
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

        let mut merged = TensorDigest::new(&[1], self.compression);
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

        let mut merged = TensorDigest::new(&[1], self.compression);
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

    /// Merge all channels of the tensor into one digest.
    pub fn merge_all(&mut self) -> Self {
        self.flush();
        let n_channels = self.numel / self.spatial_size();
        self.merge_channels(&(0..n_channels).collect::<Vec<_>>())
    }

    /// Flush pending data and write a zstd-compressed bincode snapshot to `path`.
    pub fn save(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()>
    where
        T: serde::Serialize,
    {
        self.flush();
        let bytes = bincode2::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let compressed = zstd::encode_all(bytes.as_slice(), 3).map_err(std::io::Error::other)?;
        std::fs::write(path, compressed)
    }

    /// Load and decompress a snapshot written by `save`.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        let compressed = std::fs::read(path)?;
        let bytes = zstd::decode_all(compressed.as_slice())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let loaded: Self = bincode2::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if loaded.dtype_tag != T::DTYPE_TAG {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "dtype mismatch: file contains tag {} but expected {}",
                    loaded.dtype_tag,
                    T::DTYPE_TAG
                ),
            ));
        }
        Ok(loaded)
    }

    /// Number of spatial elements per channel (product of the last two shape dims, or `numel`
    /// for tensors with fewer than two dimensions).
    fn spatial_size(&self) -> usize {
        let ndim = self.shape.len();
        if ndim < 2 {
            self.numel
        } else {
            self.shape[ndim - 2] * self.shape[ndim - 1]
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

impl<T: TensorValue> TensorDigest<T> {
    /// Launch a blocking HTTP visualizer server.
    /// Default port: 7777. Override with the `MONATQ_PORT` environment variable.
    #[cfg(feature = "visualize")]
    pub fn visualize(&mut self) -> std::io::Result<()> {
        crate::server::serve(self)
    }

    /// Launch a blocking HTTP visualizer server that exits when `stop` is set.
    #[cfg(feature = "visualize")]
    pub fn visualize_until(&mut self, stop: &AtomicBool) -> std::io::Result<()> {
        crate::server::serve_until(self, stop)
    }

    /// Return a copy with centroids centered at zero removed.
    ///
    /// This is intended for visualization of sparse tensors where exact zeros dominate the
    /// estimated density.
    pub fn without_zeros(&mut self) -> Self {
        self.flush();
        let mut filtered = TensorDigest::new(&self.shape, self.compression);
        let eps = 1e-12_f32;

        for e in 0..self.numel {
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

fn l1_simd(a: &[f32; N_PADDED], b: &[f32; N_PADDED]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    for i in (0..N_PADDED).step_by(8) {
        let va = f32x8::from(<[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        let diff = va - vb;
        acc += diff.max(-diff);
    }
    acc.reduce_add()
}

fn analyze_element(
    means: &[f32],
    weights: &[u32],
    total_weight: f32,
    min_v: f32,
    max_v: f32,
) -> Distribution {
    const D_U: f32 = 10.8;

    if means.is_empty() {
        return Distribution::Normal;
    }

    let med = quantile_from_centroids(means, weights, total_weight, min_v, max_v, 0.5);
    let std = (quantile_from_centroids(means, weights, total_weight, min_v, max_v, 0.84)
        - quantile_from_centroids(means, weights, total_weight, min_v, max_v, 0.16))
        / 2.0;

    if std.abs() < 1e-6 {
        return Distribution::Normal;
    }

    let probes = probe_points();
    let mut emp = [0f32; N_PADDED];
    for (i, &p) in probes.iter().enumerate() {
        emp[i] =
            (quantile_from_centroids(means, weights, total_weight, min_v, max_v, p) - med) / std;
    }

    let profiles = ref_profiles();
    let (best, best_dist) = Distribution::iter()
        .map(|d| (d, l1_simd(&emp, &profiles.0[d.index()])))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();

    if best_dist > D_U {
        Distribution::Unknown
    } else {
        best
    }
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
        compression as f64 / (2.0 * std::f64::consts::PI * new_total as f64 * (new_total as f64).ln())
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
    fn basic_quantile() {
        let mut td = TensorDigest::new(&[3], 100);

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
