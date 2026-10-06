use std::sync::OnceLock;

use rayon::prelude::*;

use crate::{
    Result, TensorValue,
    block::BlockLayout,
    error::{check_index, check_sample_len},
    kernels::{DigestKernel, RankKnot, RankKnotConfig, sealed},
    tensor_digest::StorageOperations,
};

pub const RANK_KNOT_K: usize = 32;

/// Knot records in the reusable merge buffer. Each record is 16 bytes, so this is 32 KiB.
///
/// One compression reads at most this many knots. [`MERGE_FANIN`] is how many summaries fit,
/// since each summary contributes at most [`RANK_KNOT_K`] knots.
const MERGE_ENTRY_BUDGET: usize = 32 * 1024 / 16;

/// Summaries combined by one compression node.
///
/// Selections wider than this are split into this many chunks. Each chunk is compressed back
/// to [`RANK_KNOT_K`] knots, then those summaries are compressed together. The longest path
/// is `ceil(log_F(M))` compressions for `F = MERGE_FANIN`.
const MERGE_FANIN: usize = MERGE_ENTRY_BUDGET / RANK_KNOT_K;

/// Leading byte of a RankKnot snapshot payload.
///
/// The crate-level [`crate::from_bytes`] loader uses this tag to identify the kernel.
/// Typed loaders also check it so a snapshot cannot be decoded with the wrong kernel.
pub(crate) const RANK_KNOT_KERNEL_TAG: u8 = 0x52;

/// Snapshot format revision. Bump whenever the field layout below changes.
const RANK_KNOT_FORMAT_VERSION: u16 = 7;

/// On-disk form of [`RankKnotStorage`].
///
/// The encoding constants travel with the data. `RANK_KNOT_K` is an unstable internal, so a
/// snapshot written by a future build with a different knot count must be rejected rather
/// than decoded into a state of the wrong width.
///
/// The generic state, layout, and weight fields let serialization borrow storage directly while
/// deserialization owns the decoded values.
#[derive(serde::Serialize, serde::Deserialize)]
struct RankKnotSnapshot<S = Vec<RankKnotState>, L = BlockLayout, W = Vec<u64>> {
    kernel_tag: u8,
    format_version: u16,
    knot_count: u32,
    dtype_tag: u8,
    sample_count: u64,
    states: S,
    layout: L,
    state_weights: W,
}

/// The leading fields of [`RankKnotSnapshot`], in the same order.
///
/// bincode encodes struct fields sequentially with no framing, so deserializing a prefix
/// struct reads exactly those fields and stops. That lets the crate-level loader identify a
/// snapshot's kernel and element type without decoding megabytes of knots first.
#[derive(serde::Deserialize)]
struct RankKnotHeader {
    kernel_tag: u8,
    format_version: u16,
    #[allow(dead_code)]
    knot_count: u32,
    dtype_tag: u8,
}

/// Identify an uncompressed payload without fully decoding it.
///
/// Returns `None` when the payload is not a RankKnot snapshot at all, so the caller can try
/// another kernel; returns `Some(dtype_tag)` when it is.
pub(crate) fn peek_dtype_tag(payload: &[u8]) -> Option<u8> {
    let header: RankKnotHeader = bincode2::deserialize(payload).ok()?;
    (header.kernel_tag == RANK_KNOT_KERNEL_TAG).then_some(header.dtype_tag)
}

fn invalid_data(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidSnapshot(message.into())
}

/// The complete persistent summary for one tensor position.
///
/// Every field is `f32` for every element type `T`. The crate-wide contract reports
/// quantiles and extrema as `f32` regardless of the input type, and the t-digest kernel
/// likewise keeps `f32` centroids for `i32` tensors, so widening this state would add cost
/// without changing a single observable answer.
#[repr(C)]
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) struct RankKnotState {
    values: [f32; RANK_KNOT_K],
    /// Raw observation counts per knot. They sum to the block's observation count unless the
    /// state was halved to fit `u32`, in which case they sum to less.
    counts: [u32; RANK_KNOT_K],
    pure_mask: u64,
    min: f32,
    max: f32,
}

impl Default for RankKnotState {
    fn default() -> Self {
        Self {
            values: [0.0; RANK_KNOT_K],
            counts: [0; RANK_KNOT_K],
            pure_mask: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    value: f32,
    weight: u64,
    pure: bool,
}

const _: () = assert!(std::mem::size_of::<Entry>() * MERGE_ENTRY_BUDGET == 32 * 1024);
const _: () = assert!(MERGE_FANIN * RANK_KNOT_K == MERGE_ENTRY_BUDGET);

struct RankKnotScratch {
    incoming: Vec<f32>,
    entries: Vec<Entry>,
    boundaries: [usize; RANK_KNOT_K - 1],
}

impl RankKnotScratch {
    fn new(rows: usize) -> Self {
        Self {
            incoming: Vec::with_capacity(rows),
            entries: Vec::with_capacity(RANK_KNOT_K + rows),
            boundaries: [0; RANK_KNOT_K - 1],
        }
    }
}

/// Per-position RankKnot summaries for one tensor.
///
/// # Element types and precision
///
/// Summary state is `f32` regardless of `T`, so an `i32` stream is summarised at `f32`
/// resolution: magnitudes above 2^24 (16,777,216) are not exactly representable and both
/// knots and extrema round to the nearest `f32`. This is not a RankKnot-specific limit. The
/// crate reports quantiles as `f32` for every element type and the t-digest kernel keeps
/// `f32` centroids for `i32` tensors too, so the ceiling is the same on either kernel.
///
/// `T` survives here only where it is genuinely load-bearing: the input buffer holds raw
/// `T` so ingestion stays a `copy_from_slice`, and the snapshot records `T::DTYPE_TAG` so an
/// `i32` snapshot cannot be loaded as `f32`.
pub(crate) struct RankKnotStorage<T> {
    layout: BlockLayout,
    config: RankKnotConfig,
    row_buffer: Vec<T>,
    buffer_rows: usize,
    n_buffered: usize,
    sample_count: u64,
    state_weights: Vec<u64>,
    states: Vec<RankKnotState>,
}

impl<T: TensorValue> RankKnotStorage<T> {
    pub(crate) fn with_config(shape: &[usize], config: RankKnotConfig) -> Self {
        Self::with_layout(BlockLayout::default_for(shape), config)
    }

    pub(crate) fn with_layout(layout: BlockLayout, config: RankKnotConfig) -> Self {
        let buffer_rows = layout.buffer_rows(config.buffer_capacity);
        let buffer_len = layout
            .input_numel()
            .checked_mul(buffer_rows)
            .expect("row buffer size overflow");
        let state_count = layout.block_count();
        Self {
            layout,
            config,
            row_buffer: vec![T::from_f32(0.0); buffer_len],
            buffer_rows,
            n_buffered: 0,
            sample_count: 0,
            state_weights: vec![0; state_count],
            states: vec![RankKnotState::default(); state_count],
        }
    }

    /// Merge row-major samples, borrowing either the input directly or the row buffer.
    fn process_batch(
        layout: &BlockLayout,
        states: &mut [RankKnotState],
        state_weights: &mut [u64],
        sample_count: &mut u64,
        data: &[T],
        rows: usize,
    ) {
        let input_numel = layout.input_numel();
        let values_per_block = rows * layout.max_block_len();
        states
            .par_iter_mut()
            .zip(state_weights.par_iter_mut())
            .with_min_len(64)
            .enumerate()
            .for_each_init(
                || RankKnotScratch::new(values_per_block),
                |scratch, (position, (state, weight))| {
                    scratch.incoming.clear();
                    let (start, stride, len) = layout.span(position);
                    for row in data.chunks_exact(input_numel) {
                        scratch
                            .incoming
                            .extend((0..len).map(|k| row[start + k * stride].to_f32()));
                    }
                    scratch
                        .incoming
                        .sort_unstable_by(|left, right| left.partial_cmp(right).unwrap());
                    let minimum = scratch.incoming[0];
                    let maximum = scratch.incoming[scratch.incoming.len() - 1];
                    let old_count = *weight;
                    merge_old_and_incoming(
                        state,
                        old_count,
                        &scratch.incoming,
                        &mut scratch.entries,
                    );
                    if old_count == 0 {
                        state.min = minimum;
                        state.max = maximum;
                    } else {
                        update_min(&mut state.min, minimum);
                        update_max(&mut state.max, maximum);
                    }
                    compress_and_store(&scratch.entries, state, &mut scratch.boundaries);
                    *weight = weight.saturating_add(scratch.incoming.len() as u64);
                },
            );
        *sample_count = sample_count.saturating_add(rows as u64);
    }

    pub(crate) fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub(crate) fn config(&self) -> &RankKnotConfig {
        &self.config
    }

    pub(crate) fn min(&mut self) -> Vec<f32> {
        self.flush();
        self.states.iter().map(|state| state.min).collect()
    }

    pub(crate) fn max(&mut self) -> Vec<f32> {
        self.flush();
        self.states.iter().map(|state| state.max).collect()
    }

    /// Callers reach this through `TensorDigest`, which validates `idx` first.
    pub(crate) fn cell_min(&mut self, idx: usize) -> f32 {
        self.flush();
        self.states[idx].min
    }

    /// Callers reach this through `TensorDigest`, which validates `idx` first.
    pub(crate) fn cell_max(&mut self, idx: usize) -> f32 {
        self.flush();
        self.states[idx].max
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

    /// Merge selected flat-indexed blocks into a single-block digest.
    ///
    /// Selections of at most [`MERGE_FANIN`] blocks are compressed in one pass. Wider selections
    /// are split into that many chunks, each chunk is compressed back to [`RANK_KNOT_K`] knots,
    /// and those summaries are compressed together. Every compression takes knot counts as
    /// population weights, sorts that support, coalesces equal values, and runs the ingestion
    /// compressor. Relative population weights survive for unequal-sized blocks, and extrema
    /// are exact unions.
    ///
    /// Scratch is one 32 KiB buffer, reused at every node. A selection of `M` blocks keeps one
    /// fan-in of partial summaries per level and recompresses `ceil(log_F(M))` times along the
    /// longest path, with `F = MERGE_FANIN`.
    pub(crate) fn merge_cells(&mut self, indices: &[usize]) -> Result<Self> {
        self.flush();
        self.merge_n(indices.len(), &|offset| indices[offset])
    }

    /// Merge every position of the selected leading-dimension channels into one digest.
    ///
    /// A "channel" is a contiguous block of `H×W` flat positions; for a 4-D tensor
    /// `[B, C, H, W]` the channel flat index is `b * C + c`. The selected blocks go through
    /// the same balanced compression as [`Self::merge_cells`].
    pub(crate) fn merge_channels(&mut self, channel_indices: &[usize]) -> Result<Self> {
        let hw = self.spatial_size();
        let mut cells = Vec::with_capacity(channel_indices.len() * hw);
        for &channel in channel_indices {
            let start = channel
                .checked_mul(hw)
                .ok_or(crate::Error::IndexOutOfBounds {
                    index: usize::MAX,
                    numel: self.layout.block_count(),
                })?;
            let end = start
                .checked_add(hw)
                .and_then(|value| value.checked_sub(1))
                .ok_or(crate::Error::IndexOutOfBounds {
                    index: usize::MAX,
                    numel: self.layout.block_count(),
                })?;
            check_index(end, self.layout.block_count())?;
            cells.extend(start..=end);
        }
        self.merge_cells(&cells)
    }

    /// Merge every tensor position into one single-position digest.
    pub(crate) fn merge_all(&mut self) -> Result<Self> {
        self.flush();
        let count = self.layout.block_count();
        self.merge_n(count, &|offset| offset)
    }

    /// Balanced merge of `len` blocks addressed by `index_of`.
    ///
    /// `index_of` maps a dense offset in `0..len` to a flat block index. `merge_all` uses the
    /// identity so it does not allocate an index vector; `merge_cells` passes the caller's
    /// selection.
    fn merge_n(&self, len: usize, index_of: &impl Fn(usize) -> usize) -> Result<Self> {
        let mut merged = RankKnotStorage::with_config(&[1], self.config);
        if len == 0 {
            return Ok(merged);
        }
        let mut entries = Vec::with_capacity(MERGE_ENTRY_BUDGET);
        let mut boundaries = [0_usize; RANK_KNOT_K - 1];
        let (state, weight) = self.merge_range(0, len, index_of, &mut entries, &mut boundaries)?;
        // A scalar merged digest receives one observation per represented source observation,
        // so future scalar updates continue with the same old/new weighting.
        merged.states[0] = state;
        merged.state_weights[0] = weight;
        merged.sample_count = weight;
        Ok(merged)
    }

    fn merge_range(
        &self,
        base: usize,
        len: usize,
        index_of: &impl Fn(usize) -> usize,
        entries: &mut Vec<Entry>,
        boundaries: &mut [usize; RANK_KNOT_K - 1],
    ) -> Result<(RankKnotState, u64)> {
        let mut parts = [(RankKnotState::default(), 0_u64); MERGE_FANIN];
        if len <= MERGE_FANIN {
            for (offset, slot) in parts.iter_mut().enumerate().take(len) {
                let index = index_of(base + offset);
                check_index(index, self.layout.block_count())?;
                *slot = (self.states[index], self.state_weights[index]);
            }
            return Ok(compress_support(&parts[..len], entries, boundaries));
        }

        // As-even chunks, with the extra blocks given to the earlier chunks. Every chunk is
        // shorter than `len`, so the recursion terminates.
        let mut cursor = 0;
        for (slot, part) in parts.iter_mut().enumerate() {
            let slots_left = MERGE_FANIN - slot;
            let take = (len - cursor).div_ceil(slots_left);
            *part = self.merge_range(base + cursor, take, index_of, entries, boundaries)?;
            cursor += take;
        }
        debug_assert_eq!(cursor, len);
        Ok(compress_support(&parts, entries, boundaries))
    }

    /// Return a copy with knots sitting at zero removed.
    ///
    /// Intended for inspecting sparse tensors where exact zeros dominate the density and
    /// hide the shape of everything else.
    ///
    /// The surviving knots keep their counts and are re-run through the shared compressor.
    /// Quantiles of the result are therefore quantiles
    /// *of the nonzero subpopulation*, not of the original stream.
    ///
    /// One consequence is worth stating plainly: `sample_count` is storage-wide in RankKnot,
    /// not per position, so it cannot be reduced to "the number of nonzero observations" —
    /// that count differs per position. It is carried over unchanged and no longer agrees
    /// with the filtered distribution. Treat a filtered digest as a shape to look at, not as
    /// a population to count.
    pub(crate) fn without_zeros(&mut self) -> Self {
        self.flush();
        let mut filtered = RankKnotStorage::with_layout(self.layout.clone(), self.config);
        filtered.sample_count = self.sample_count;
        filtered.state_weights.clone_from(&self.state_weights);

        let mut entries = Vec::with_capacity(RANK_KNOT_K);
        let mut boundaries = [0_usize; RANK_KNOT_K - 1];
        for (position, state) in self.states.iter().enumerate() {
            entries.clear();
            for index in 0..RANK_KNOT_K {
                let count = state.counts[index];
                if count == 0 || !state.values[index].is_nonzero() {
                    continue;
                }
                entries.push(Entry {
                    value: state.values[index],
                    weight: u64::from(count),
                    pure: state.pure_mask & (1_u64 << index) != 0,
                });
            }

            let target = &mut filtered.states[position];
            if entries.is_empty() {
                // Nothing survived. Leave the default empty state, whose queries answer 0.0,
                // and pin the extrema to zero rather than leaving the ±inf sentinels.
                target.min = 0.0;
                target.max = 0.0;
                continue;
            }

            // Prefer the original extremum when it was not itself a zero, so filtering does
            // not pull the reported range in to the outermost surviving knot.
            target.min = if state.min.is_nonzero() {
                state.min
            } else {
                entries[0].value
            };
            target.max = if state.max.is_nonzero() {
                state.max
            } else {
                entries[entries.len() - 1].value
            };
            compress_and_store(&entries, target, &mut boundaries);
        }
        filtered
    }

    /// Flush pending rows and return a zstd-compressed bincode snapshot.
    ///
    /// Only the compressed summary state is persisted. Buffered rows are folded in by the
    /// flush, and `buffer_capacity` is deliberately *not* stored: it is an ingestion tuning
    /// knob that does not affect the encoded distribution, so a loaded digest starts from
    /// [`RankKnotConfig::default`] and callers who care must reapply their own value.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_bytes(&mut self) -> Result<Vec<u8>> {
        self.flush();
        let snapshot = RankKnotSnapshot {
            kernel_tag: RANK_KNOT_KERNEL_TAG,
            format_version: RANK_KNOT_FORMAT_VERSION,
            knot_count: RANK_KNOT_K as u32,
            dtype_tag: T::DTYPE_TAG,
            sample_count: self.sample_count,
            states: self.states.as_slice(),
            layout: &self.layout,
            state_weights: self.state_weights.as_slice(),
        };
        let payload =
            bincode2::serialize(&snapshot).map_err(|error| invalid_data(error.to_string()))?;
        zstd::encode_all(payload.as_slice(), 3).map_err(crate::Error::Io)
    }

    /// Load and decompress a snapshot from memory.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let payload = zstd::decode_all(bytes).map_err(|error| invalid_data(error.to_string()))?;
        Self::from_payload(&payload)
    }

    /// Decode an uncompressed snapshot payload.
    ///
    /// Every field that the reader relies on is validated before it is used. The kernel and
    /// version header is checked before the full state is decoded. Inconsistent lengths or
    /// counts are rejected before query use.
    pub(crate) fn from_payload(payload: &[u8]) -> Result<Self> {
        let header: RankKnotHeader =
            bincode2::deserialize(payload).map_err(|error| invalid_data(error.to_string()))?;
        if header.kernel_tag != RANK_KNOT_KERNEL_TAG {
            return Err(invalid_data(format!(
                "not a RankKnot snapshot: kernel tag {} but expected {RANK_KNOT_KERNEL_TAG}",
                header.kernel_tag
            )));
        }
        if header.format_version != RANK_KNOT_FORMAT_VERSION {
            return Err(invalid_data(format!(
                "unsupported RankKnot snapshot version {} but expected {RANK_KNOT_FORMAT_VERSION}",
                header.format_version
            )));
        }
        let snapshot: RankKnotSnapshot =
            bincode2::deserialize(payload).map_err(|error| invalid_data(error.to_string()))?;

        if snapshot.knot_count as usize != RANK_KNOT_K {
            return Err(invalid_data(format!(
                "snapshot encoding mismatch: K={} but this build uses K={RANK_KNOT_K}",
                snapshot.knot_count
            )));
        }
        let expected_tag = T::DTYPE_TAG;
        if snapshot.dtype_tag != expected_tag {
            return Err(invalid_data(format!(
                "dtype mismatch: snapshot contains tag {} but expected {expected_tag}",
                snapshot.dtype_tag
            )));
        }

        snapshot.layout.validate()?;
        let state_count = snapshot.layout.block_count();
        if snapshot.states.len() != state_count || snapshot.state_weights.len() != state_count {
            return Err(invalid_data(
                "snapshot arrays do not match the stored shape",
            ));
        }
        for (position, (state, &weight)) in snapshot
            .states
            .iter()
            .zip(&snapshot.state_weights)
            .enumerate()
        {
            validate_state(state, weight, position)?;
        }

        let mut storage = Self::with_layout(snapshot.layout, RankKnotConfig::default());
        storage.sample_count = snapshot.sample_count;
        storage.state_weights = snapshot.state_weights;
        storage.states = snapshot.states;
        Ok(storage)
    }

    /// Move summaries when the compact block order changes. Pending rows stay put: they
    /// are stored in flat element order, and a legal retarget keeps each block's elements.
    fn apply_layout(&mut self, change: crate::block::LayoutChange) {
        if let Some(order) = &change.new_to_old {
            let states = order.iter().map(|&old| self.states[old]).collect();
            let weights = order.iter().map(|&old| self.state_weights[old]).collect();
            self.states = states;
            self.state_weights = weights;
        }
        if let Some(order) = &change.element_order {
            crate::block::permute_rows(&mut self.row_buffer, self.n_buffered, order);
        }
        debug_assert_eq!(
            change.layout.buffer_rows(self.config.buffer_capacity),
            self.buffer_rows
        );
        self.layout = change.layout;
    }
}

/// Reject a decoded state that violates an invariant the query path assumes.
///
/// Queries stay memory-safe on any input, so these checks exist to prevent a corrupt or
/// hand-edited snapshot from silently producing wrong quantiles.
///
/// Counts never sum to more than the observation weight: they are equal until halving or
/// zero filtering removes population, and both only ever lower the sum (filtering can take
/// it to zero).
fn validate_state(state: &RankKnotState, weight: u64, position: usize) -> Result<()> {
    let total = count_total(state);
    if total > weight {
        return Err(invalid_data(format!(
            "position {position}: knot counts sum to {total}, inconsistent with observation weight {weight}"
        )));
    }

    let mut previous: Option<f32> = None;
    for index in 0..RANK_KNOT_K {
        if state.counts[index] == 0 {
            continue;
        }
        let value = state.values[index];
        if value.is_nan() {
            return Err(invalid_data(format!(
                "position {position}: knot {index} is NaN, which RankKnot does not support"
            )));
        }
        if previous.is_some_and(|previous| previous.total_cmp(&value).is_gt()) {
            return Err(invalid_data(format!(
                "position {position}: knot values are not in ascending order at index {index}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn count_total(state: &RankKnotState) -> u64 {
    state.counts.iter().map(|&count| u64::from(count)).sum()
}

/// The population weight a knot represents when merged with new support.
///
/// Exactly `count` unless the state was halved to fit `u32`; then the count is scaled back up
/// by `observation_weight / count_total` so old knots are not underweighted against new data.
/// Floor division keeps the rescaled sum at or below `observation_weight`, and since
/// `observation_weight >= count_total` the result is never below `count`.
fn population_weight(count: u32, observation_weight: u64, count_total: u64) -> u64 {
    if observation_weight <= count_total {
        return u64::from(count);
    }
    (u128::from(count) * u128::from(observation_weight) / u128::from(count_total)) as u64
}

impl<T: TensorValue> sealed::Kernel<T> for RankKnot {
    type Storage = RankKnotStorage<T>;

    fn create_storage(shape: &[usize], config: <Self as DigestKernel<T>>::Config) -> Self::Storage {
        RankKnotStorage::with_config(shape, config)
    }

    fn create_storage_with_layout(
        layout: BlockLayout,
        config: <Self as DigestKernel<T>>::Config,
    ) -> Self::Storage {
        RankKnotStorage::with_layout(layout, config)
    }
}

impl<T: TensorValue> DigestKernel<T> for RankKnot {
    type Config = RankKnotConfig;
}

impl<T: TensorValue> StorageOperations<T> for RankKnotStorage<T> {
    fn shape(&self) -> &[usize] {
        self.layout.input_shape()
    }
    fn numel(&self) -> usize {
        self.layout.input_numel()
    }
    fn block_shape(&self) -> &[usize] {
        self.layout.shape()
    }
    fn block_count(&self) -> usize {
        self.layout.block_count()
    }
    fn block_config(&self) -> crate::BlockConfig {
        self.layout.config()
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        let change = self
            .layout
            .retarget(shape, self.layout.config(), "reshape")?;
        self.apply_layout(change);
        Ok(())
    }

    fn remap(&mut self, shape: &[usize], blocks: crate::BlockConfig) -> Result<()> {
        let change = self.layout.retarget(shape, blocks, "remap")?;
        self.apply_layout(change);
        Ok(())
    }

    fn permute(&mut self, axes: &[isize]) -> Result<()> {
        let change = self.layout.permute(axes)?;
        self.apply_layout(change);
        Ok(())
    }

    fn total_weight(&self, idx: usize) -> Result<u32> {
        check_index(idx, self.layout.block_count())?;
        Ok(self.state_weights[idx].min(u32::MAX as u64) as u32)
    }

    fn update(&mut self, data: &[T]) -> Result<()> {
        let input_numel = self.layout.input_numel();
        check_sample_len(data.len(), input_numel)?;
        if self.buffer_rows == 0 {
            Self::process_batch(
                &self.layout,
                &mut self.states,
                &mut self.state_weights,
                &mut self.sample_count,
                data,
                1,
            );
            return Ok(());
        }
        let start = self.n_buffered * input_numel;
        self.row_buffer[start..start + input_numel].copy_from_slice(data);
        self.n_buffered += 1;
        if self.n_buffered == self.buffer_rows {
            self.flush();
        }
        Ok(())
    }

    fn flush(&mut self) {
        if self.n_buffered == 0 {
            return;
        }
        Self::process_batch(
            &self.layout,
            &mut self.states,
            &mut self.state_weights,
            &mut self.sample_count,
            &self.row_buffer[..self.n_buffered * self.layout.input_numel()],
            self.n_buffered,
        );
        self.n_buffered = 0;
    }

    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.flush();
        self.states
            .par_iter()
            .map(|state| query(state, q))
            .collect()
    }

    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        self.flush();
        qs.iter()
            .map(|&q| {
                self.states
                    .par_iter()
                    .map(|state| query(state, q))
                    .collect()
            })
            .collect()
    }

    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Result<Vec<f32>> {
        check_index(idx, self.layout.block_count())?;
        self.flush();
        Ok(qs.iter().map(|&q| query(&self.states[idx], q)).collect())
    }

    fn merge_cells(&mut self, indices: &[usize]) -> Result<Self> {
        RankKnotStorage::merge_cells(self, indices)
    }

    fn merge_channels(&mut self, channel_indices: &[usize]) -> Result<Self> {
        RankKnotStorage::merge_channels(self, channel_indices)
    }

    fn merge_all(&mut self) -> Result<Self> {
        RankKnotStorage::merge_all(self)
    }

    fn analyze(&mut self) -> Result<Vec<crate::Distribution>> {
        self.flush();
        Ok(self
            .states
            .par_iter()
            .map(|state| {
                if state.counts[0] == 0 {
                    return crate::Distribution::Unknown;
                }
                crate::distribution::classify(|q| query(state, q))
            })
            .collect())
    }

    fn without_zeros(&mut self) -> Result<Self> {
        Ok(RankKnotStorage::without_zeros(self))
    }

    fn to_bytes(&mut self) -> Result<Vec<u8>>
    where
        T: serde::Serialize,
    {
        RankKnotStorage::to_bytes(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        RankKnotStorage::from_bytes(bytes)
    }

    fn from_payload(payload: &[u8]) -> Result<Self>
    where
        T: serde::de::DeserializeOwned,
    {
        RankKnotStorage::from_payload(payload)
    }

    #[cfg(feature = "visualize")]
    fn visualize(&mut self) -> Result<()> {
        crate::server::serve(self)
    }

    #[cfg(feature = "visualize")]
    fn visualize_until(&mut self, stop: &std::sync::atomic::AtomicBool) -> Result<()> {
        crate::server::serve_until(self, stop)
    }
}

fn merge_old_and_incoming(
    state: &RankKnotState,
    old_count: u64,
    incoming: &[f32],
    output: &mut Vec<Entry>,
) {
    output.clear();
    if old_count == 0 {
        for &value in incoming {
            push_coalesced(
                output,
                Entry {
                    value,
                    weight: 1,
                    pure: true,
                },
            );
        }
        return;
    }
    let count_total = count_total(state);
    let mut old_index = 0;
    let mut new_index = 0;
    while old_index < RANK_KNOT_K || new_index < incoming.len() {
        while old_index < RANK_KNOT_K && state.counts[old_index] == 0 {
            old_index += 1;
        }
        let take_old = old_index < RANK_KNOT_K
            && (new_index == incoming.len()
                || state.values[old_index]
                    .total_cmp(&incoming[new_index])
                    .is_le());
        if take_old {
            push_coalesced(
                output,
                Entry {
                    value: state.values[old_index],
                    weight: population_weight(state.counts[old_index], old_count, count_total),
                    pure: state.pure_mask & (1_u64 << old_index) != 0,
                },
            );
            old_index += 1;
        } else if new_index < incoming.len() {
            push_coalesced(
                output,
                Entry {
                    value: incoming[new_index],
                    weight: 1,
                    pure: true,
                },
            );
            new_index += 1;
        } else {
            break;
        }
    }
}

fn push_coalesced(entries: &mut Vec<Entry>, entry: Entry) {
    if let Some(last) = entries.last_mut()
        && last.value == entry.value
    {
        last.weight = last.weight.saturating_add(entry.weight);
        last.pure &= entry.pure;
    } else {
        entries.push(entry);
    }
}

/// Union `parts` into one summary.
///
/// `entries` and `boundaries` are caller-owned scratch reused by every node in the tree.
/// Each part contributes its knots at population weight, so a partial that was halved to
/// fit `u32` is scaled back up by its observation count before the next compression.
fn compress_support(
    parts: &[(RankKnotState, u64)],
    entries: &mut Vec<Entry>,
    boundaries: &mut [usize; RANK_KNOT_K - 1],
) -> (RankKnotState, u64) {
    entries.clear();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut weight = 0_u64;
    for &(state, observation_weight) in parts {
        update_min(&mut min, state.min);
        update_max(&mut max, state.max);
        weight = weight.saturating_add(observation_weight);
        let total = count_total(&state);
        for index in 0..RANK_KNOT_K {
            let count = state.counts[index];
            if count == 0 {
                continue;
            }
            entries.push(Entry {
                value: state.values[index],
                weight: population_weight(count, observation_weight, total),
                pure: state.pure_mask & (1_u64 << index) != 0,
            });
        }
    }

    let mut state = RankKnotState {
        min,
        max,
        ..RankKnotState::default()
    };
    if entries.is_empty() {
        return (state, weight);
    }

    // Sort once, then coalesce equal values in place. `dedup_by` hands out the kept
    // predecessor by mutable reference, so the union needs no second buffer.
    entries.sort_unstable_by(|left, right| left.value.total_cmp(&right.value));
    entries.dedup_by(|entry, kept| {
        if kept.value == entry.value {
            kept.weight = kept.weight.saturating_add(entry.weight);
            kept.pure &= entry.pure;
            true
        } else {
            false
        }
    });
    compress_and_store(entries, &mut state, boundaries);
    (state, weight)
}

fn compress_and_store(
    entries: &[Entry],
    state: &mut RankKnotState,
    boundaries: &mut [usize; RANK_KNOT_K - 1],
) {
    let (total, boundary_count) = make_boundaries(entries, boundaries);
    *state = RankKnotState {
        min: state.min,
        max: state.max,
        ..RankKnotState::default()
    };
    if total == 0 {
        // No weighted support. The reset state above is already the correct empty summary.
        return;
    }

    let mut start = 0;
    let mut pending: Option<Entry> = None;
    let mut counts = [0_u64; RANK_KNOT_K];
    let mut output_index = 0;
    for end in boundaries[..boundary_count]
        .iter()
        .copied()
        .chain(std::iter::once(entries.len()))
    {
        let pure = end == start + 1 && entries[start].pure;
        let (mass, value) = if pure {
            (entries[start].weight, entries[start].value)
        } else {
            let (mass, moment) =
                entries[start..end]
                    .iter()
                    .fold((0_u64, 0.0_f64), |(mass, moment), entry| {
                        (
                            mass.saturating_add(entry.weight),
                            moment + entry.value as f64 * entry.weight as f64,
                        )
                    });
            let value = if entries[start].value.is_finite() {
                (moment / mass as f64) as f32
            } else {
                entries[start].value
            };
            (mass, value)
        };
        debug_assert!(value.is_finite() || pure);
        let representative = Entry {
            value,
            weight: mass,
            pure,
        };
        if let Some(previous) = pending.as_mut()
            && previous.value == representative.value
        {
            previous.weight = previous.weight.saturating_add(representative.weight);
            previous.pure &= representative.pure;
        } else if let Some(previous) = pending.replace(representative) {
            store_representative(state, &mut counts, previous, &mut output_index);
        }
        start = end;
    }
    if let Some(previous) = pending {
        store_representative(state, &mut counts, previous, &mut output_index);
    }

    // Halve every count, rounding up so no active knot reaches zero, until the largest fits.
    let largest = counts.iter().copied().max().unwrap_or(0);
    let mut shift = 0;
    while largest.div_ceil(1_u64 << shift) > u64::from(u32::MAX) {
        shift += 1;
    }
    for (slot, &count) in state.counts.iter_mut().zip(&counts) {
        *slot = count.div_ceil(1_u64 << shift) as u32;
    }
}

fn store_representative(
    state: &mut RankKnotState,
    counts: &mut [u64; RANK_KNOT_K],
    representative: Entry,
    output_index: &mut usize,
) {
    debug_assert!(representative.weight != 0);
    state.values[*output_index] = representative.value;
    counts[*output_index] = representative.weight;
    if representative.pure {
        state.pure_mask |= 1_u64 << *output_index;
    }
    *output_index += 1;
}

fn make_boundaries(entries: &[Entry], boundaries: &mut [usize; RANK_KNOT_K - 1]) -> (u64, usize) {
    let total = entries.iter().map(|entry| entry.weight).sum();
    if entries.len() <= RANK_KNOT_K {
        for (slot, boundary) in boundaries.iter_mut().zip(1..entries.len()) {
            *slot = boundary;
        }
        return (total, entries.len() - 1);
    }
    let mut boundary_count = 0;
    // Infinities are indivisible singleton groups and must never enter a mean.
    // Targets are ordered, so boundaries can be emitted in order without sorting.
    let positive_infinity = entries
        .last()
        .is_some_and(|entry| entry.value == f32::INFINITY);
    if entries
        .first()
        .is_some_and(|entry| entry.value == f32::NEG_INFINITY)
    {
        boundaries[boundary_count] = 1;
        boundary_count += 1;
    }
    let target_limit = RANK_KNOT_K - 1 - usize::from(positive_infinity);
    let mut index = 0;
    let mut before = 0_u64;
    let mut after = entries[0].weight;
    for &q in arcsine_targets() {
        let target = q * total as f64;
        while index + 1 < entries.len() && (after as f64) < target {
            before = after;
            index += 1;
            after = after.saturating_add(entries[index].weight);
        }
        let before = before as f64;
        let after = after as f64;
        let boundary = if target - before <= after - target {
            index
        } else {
            index + 1
        };
        if boundary > 0
            && boundary < entries.len()
            && (boundary_count == 0 || boundaries[boundary_count - 1] != boundary)
            && boundary_count < target_limit
        {
            boundaries[boundary_count] = boundary;
            boundary_count += 1;
        }
    }
    if positive_infinity
        && (boundary_count == 0 || boundaries[boundary_count - 1] != entries.len() - 1)
    {
        boundaries[boundary_count] = entries.len() - 1;
        boundary_count += 1;
    }
    (total, boundary_count)
}

fn arcsine_targets() -> &'static [f64; RANK_KNOT_K - 1] {
    static TARGETS: OnceLock<[f64; RANK_KNOT_K - 1]> = OnceLock::new();
    TARGETS.get_or_init(|| {
        std::array::from_fn(|index| {
            let cut = index + 1;
            (std::f64::consts::PI * cut as f64 / (2.0 * RANK_KNOT_K as f64))
                .sin()
                .powi(2)
        })
    })
}

fn query(state: &RankKnotState, q: f32) -> f32 {
    if state.counts[0] == 0 {
        return 0.0;
    }
    if q.is_nan() {
        return f32::NAN;
    }
    if q <= 0.0 {
        return state.min;
    }
    if q >= 1.0 {
        return state.max;
    }
    let total = count_total(state) as f64;
    let target = q as f64 * total;
    let first = state.counts.iter().position(|&count| count != 0).unwrap();
    let mut previous_rank = 0.0_f64;
    let mut previous_value = state.values[first];
    let mut cumulative = 0_u64;
    for index in 0..RANK_KNOT_K {
        let count = u64::from(state.counts[index]);
        if count == 0 {
            continue;
        }
        let left = cumulative as f64;
        cumulative += count;
        let right = cumulative as f64;
        let value = state.values[index];
        if state.pure_mask & (1_u64 << index) != 0 {
            if target >= left && target <= right {
                return value;
            }
            if target <= left {
                return interpolate(previous_rank, previous_value, left, value, target);
            }
            previous_rank = right;
            previous_value = value;
        } else {
            let rank = (left + right) * 0.5;
            if target <= rank {
                return interpolate(previous_rank, previous_value, rank, value, target);
            }
            previous_rank = rank;
            previous_value = value;
        }
    }
    interpolate(
        previous_rank,
        previous_value,
        total,
        state.values[first_active_from_end(state)],
        target,
    )
}

fn first_active_from_end(state: &RankKnotState) -> usize {
    state.counts.iter().rposition(|&count| count != 0).unwrap()
}

fn interpolate(left_q: f64, left: f32, right_q: f64, right: f32, q: f64) -> f32 {
    if left == right || right_q <= left_q {
        return left;
    }
    if !left.is_finite() {
        return left;
    }
    if !right.is_finite() {
        return right;
    }
    let fraction = ((q - left_q) / (right_q - left_q)).clamp(0.0, 1.0);
    (left as f64 + (right as f64 - left as f64) * fraction) as f32
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_mode_never_allocates_or_uses_tensor_row_buffer() {
        let layout =
            BlockLayout::new(&[4, 1024], crate::BlockConfig::blocks_per_axis(4, 1)).unwrap();
        let mut storage = RankKnotStorage::<f32>::with_layout(
            layout,
            RankKnotConfig {
                buffer_capacity: 256,
            },
        );
        assert!(storage.row_buffer.is_empty());
        storage.update(&vec![1.0; 4096]).unwrap();
        assert!(storage.row_buffer.is_empty());
        assert_eq!(storage.n_buffered, 0);
    }

    #[test]
    fn small_blocks_buffer_enough_rows_to_fill_capacity() {
        let layout = BlockLayout::new(&[2, 8], crate::BlockConfig::block_size(4, 1)).unwrap();
        let mut storage =
            RankKnotStorage::<f32>::with_layout(layout, RankKnotConfig { buffer_capacity: 8 });
        assert_eq!(storage.buffer_rows, 2);
        assert_eq!(storage.row_buffer.len(), 32);
        storage.update(&[1.0; 16]).unwrap();
        assert_eq!(storage.n_buffered, 1);
        assert_eq!(storage.state_weights, vec![0; 4]);
        storage.update(&[2.0; 16]).unwrap();
        assert_eq!(storage.n_buffered, 0);
        assert_eq!(storage.state_weights, vec![8; 4]);
    }

    #[test]
    fn zero_capacity_disables_buffer_for_blocks() {
        let layout = BlockLayout::new(&[2, 8], crate::BlockConfig::block_size(4, 1)).unwrap();
        let mut storage =
            RankKnotStorage::<f32>::with_layout(layout, RankKnotConfig { buffer_capacity: 0 });
        assert!(storage.row_buffer.is_empty());
        storage.update(&[1.0; 16]).unwrap();
        assert_eq!(storage.state_weights, vec![4; 4]);
    }

    #[test]
    fn zero_capacity_updates_immediately_without_buffering() {
        let mut direct =
            RankKnotStorage::<f32>::with_config(&[2], RankKnotConfig { buffer_capacity: 0 });
        let mut single =
            RankKnotStorage::<f32>::with_config(&[2], RankKnotConfig { buffer_capacity: 1 });
        for i in 0..100 {
            let sample = [i as f32, -(i as f32)];
            direct.update(&sample).unwrap();
            single.update(&sample).unwrap();
            assert_eq!(direct.sample_count, i as u64 + 1);
            assert_eq!(direct.n_buffered, 0);
            assert!(direct.row_buffer.is_empty());
            for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
                assert_eq!(direct.quantile(q), single.quantile(q));
            }
        }
        assert!(direct.update(&[1.0]).is_err());
        assert_eq!(direct.sample_count, 100);
        direct.flush();
        assert_eq!(direct.sample_count, 100);
    }

    /// Build a valid snapshot payload, let `tamper` corrupt it, and assert the loader
    /// refuses it. Snapshots are untrusted input, so every guard needs direct coverage.
    fn assert_rejected(tamper: impl FnOnce(&mut RankKnotSnapshot), expected_fragment: &str) {
        let mut storage = RankKnotStorage::with_config(&[2], RankKnotConfig::default());
        for index in 0..1_000 {
            storage.update(&[index as f32, -(index as f32)]).unwrap();
        }
        let bytes = storage.to_bytes().expect("serialization failed");
        let payload = zstd::decode_all(bytes.as_slice()).expect("decode failed");
        let mut snapshot: RankKnotSnapshot =
            bincode2::deserialize(&payload).expect("valid snapshot must parse");

        tamper(&mut snapshot);

        let tampered = bincode2::serialize(&snapshot).expect("reserialization failed");
        let Err(error) = RankKnotStorage::<f32>::from_payload(&tampered) else {
            panic!("tampered snapshot must be rejected");
        };
        assert!(error.is_invalid_snapshot(), "unexpected error: {error}");
        assert!(
            error.to_string().contains(expected_fragment),
            "error {error:?} does not mention {expected_fragment:?}"
        );
    }

    #[test]
    fn snapshot_guards_reject_corrupt_state() {
        assert_rejected(
            |snapshot| snapshot.kernel_tag ^= 0xff,
            "not a RankKnot snapshot",
        );
        assert_rejected(
            |snapshot| snapshot.format_version += 1,
            "unsupported RankKnot snapshot version",
        );
        assert_rejected(|snapshot| snapshot.knot_count = 64, "encoding mismatch");
        assert_rejected(|snapshot| snapshot.dtype_tag = 1, "dtype mismatch");
        assert_rejected(
            |snapshot| snapshot.states.pop().map(|_| ()).unwrap(),
            "do not match the stored shape",
        );
        assert_rejected(
            |snapshot| {
                snapshot.state_weights.pop();
            },
            "do not match the stored shape",
        );
        // Counts exceeding the observation weight would answer every query with a wrong rank.
        assert_rejected(
            |snapshot| {
                let state = &mut snapshot.states[0];
                let index = state.counts.iter().position(|&count| count != 0).unwrap();
                state.counts[index] += 1;
            },
            "inconsistent with observation weight",
        );
        assert_rejected(
            |snapshot| snapshot.state_weights[0] = 0,
            "inconsistent with observation weight",
        );
        // Out-of-order or NaN knots would break the monotone decode.
        assert_rejected(
            |snapshot| {
                let state = &mut snapshot.states[0];
                let active = (0..RANK_KNOT_K)
                    .filter(|&index| state.counts[index] != 0)
                    .collect::<Vec<_>>();
                state.values.swap(active[0], active[active.len() - 1]);
            },
            "not in ascending order",
        );
        assert_rejected(
            |snapshot| {
                let state = &mut snapshot.states[0];
                let index = (0..RANK_KNOT_K)
                    .find(|&index| state.counts[index] != 0)
                    .unwrap();
                state.values[index] = f32::NAN;
            },
            "NaN",
        );
    }

    #[test]
    fn merged_state_keeps_the_encoding_invariants() {
        let mut storage = RankKnotStorage::with_config(&[4], RankKnotConfig::default());
        for index in 0..5_000 {
            let value = (index % 331) as f32;
            storage
                .update(&[value, -value, 3.0, f32::INFINITY])
                .unwrap();
        }
        let merged = storage.merge_all().unwrap();
        let state = &merged.states[0];
        assert_eq!(merged.sample_count(), 20_000);
        assert_eq!(count_total(state), 20_000);
        assert_eq!(merged.state_weights[0], 20_000);
        let active = (0..RANK_KNOT_K)
            .filter(|&index| state.counts[index] != 0)
            .collect::<Vec<_>>();
        assert!(active.windows(2).all(|pair| {
            state.values[pair[0]]
                .total_cmp(&state.values[pair[1]])
                .is_le()
        }));
        for &index in &active {
            if !state.values[index].is_finite() {
                assert_ne!(state.pure_mask & (1_u64 << index), 0);
            }
        }
        assert_eq!(state.min, -330.0);
        assert_eq!(state.max, f32::INFINITY);
    }

    #[test]
    fn reshape_keeps_pending_rows_until_a_later_flush() {
        let mut storage =
            RankKnotStorage::<f32>::with_config(&[2, 3], RankKnotConfig { buffer_capacity: 8 });
        let sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        storage.update(&sample).unwrap();
        assert_eq!(storage.n_buffered, 1);
        assert!(storage.state_weights.iter().all(|&weight| weight == 0));
        let pending = storage.row_buffer.clone();

        StorageOperations::reshape(&mut storage, &[3, 2]).unwrap();
        assert_eq!(storage.layout.input_shape(), &[3, 2]);
        assert_eq!(storage.n_buffered, 1);
        assert_eq!(storage.row_buffer, pending);
        assert!(storage.state_weights.iter().all(|&weight| weight == 0));

        let mut blocked = RankKnotStorage::<f32>::with_layout(
            BlockLayout::new(&[2, 8], crate::BlockConfig::block_size(4, -1)).unwrap(),
            RankKnotConfig { buffer_capacity: 8 },
        );
        let pooled: Vec<f32> = (0..16).map(|value| value as f32).collect();
        blocked.update(&pooled).unwrap();
        assert_eq!(blocked.n_buffered, 1);
        let pooled_pending = blocked.row_buffer.clone();
        let weights = blocked.state_weights.clone();
        StorageOperations::reshape(&mut blocked, &[4, 4]).unwrap();
        assert_eq!(blocked.layout.input_shape(), &[4, 4]);
        assert_eq!(blocked.layout.shape(), &[4, 1]);
        assert_eq!(blocked.n_buffered, 1);
        assert_eq!(blocked.row_buffer, pooled_pending);
        assert_eq!(blocked.state_weights, weights);

        let error = StorageOperations::reshape(&mut blocked, &[8, 2]).unwrap_err();
        assert!(error.is_incompatible_layout());
        assert_eq!(blocked.layout.input_shape(), &[4, 4]);
        assert_eq!(blocked.n_buffered, 1);
        assert_eq!(blocked.row_buffer, pooled_pending);
        assert_eq!(blocked.state_weights, weights);
    }

    #[test]
    fn apply_layout_permutes_summaries_without_moving_pending_rows() {
        let mut storage =
            RankKnotStorage::<f32>::with_config(&[4], RankKnotConfig { buffer_capacity: 4 });
        storage.update(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let pending = storage.row_buffer.clone();
        let buffered = storage.n_buffered;
        for block in 0..4 {
            storage.state_weights[block] = block as u64 + 1;
            storage.states[block].min = block as f32;
            storage.states[block].max = block as f32 + 10.0;
        }
        // Current one-axis layouts keep identity order. This installs a swapped order directly
        // so a future retarget that does permute still leaves the flat row buffer alone.
        storage.apply_layout(crate::block::LayoutChange {
            layout: BlockLayout::default_for(&[2, 2]),
            new_to_old: Some(vec![3, 1, 2, 0]),
            element_order: None,
        });
        assert_eq!(storage.n_buffered, buffered);
        assert_eq!(storage.row_buffer, pending);
        assert_eq!(storage.state_weights, vec![4, 2, 3, 1]);
        assert_eq!(
            storage
                .states
                .iter()
                .map(|state| state.min)
                .collect::<Vec<_>>(),
            vec![3.0, 1.0, 2.0, 0.0]
        );
        assert_eq!(
            storage
                .states
                .iter()
                .map(|state| state.max)
                .collect::<Vec<_>>(),
            vec![13.0, 11.0, 12.0, 10.0]
        );
        assert_eq!(storage.layout.input_shape(), &[2, 2]);
    }

    #[test]
    fn overflowing_counts_are_halved_without_dropping_knots() {
        let mut storage =
            RankKnotStorage::<f32>::with_config(&[1], RankKnotConfig { buffer_capacity: 0 });
        let state = &mut storage.states[0];
        state.values[0] = 1.0;
        state.values[1] = 2.0;
        state.counts[0] = u32::MAX;
        state.counts[1] = 1;
        state.pure_mask = 0b11;
        state.min = 1.0;
        state.max = 2.0;
        storage.state_weights[0] = u64::from(u32::MAX) + 1;
        storage.update(&[1.0]).unwrap();

        let state = &storage.states[0];
        assert_eq!(state.counts[0], (u32::MAX / 2) + 1);
        assert_eq!(state.counts[1], 1);
        assert_eq!(storage.state_weights[0], u64::from(u32::MAX) + 2);
        assert!(count_total(state) <= storage.state_weights[0]);

        // The halved state is rescaled on the next merge, so new data is not overweighted.
        storage.update(&[2.0]).unwrap();
        let state = &storage.states[0];
        assert_eq!(state.counts[0], u32::MAX);
        assert_eq!(state.counts[1], 2);
        assert!(count_total(state) <= storage.state_weights[0]);
        let bytes = storage.to_bytes().unwrap();
        RankKnotStorage::<f32>::from_bytes(&bytes).unwrap();
    }

    #[test]
    fn encoded_state_invariants_hold() {
        let mut storage = RankKnotStorage::with_config(
            &[1],
            RankKnotConfig {
                buffer_capacity: 17,
            },
        );
        for index in 0..2_003 {
            let value = match index {
                0 => f32::NEG_INFINITY,
                1 => f32::INFINITY,
                _ if index % 11 == 0 => 3.0,
                _ => ((index * 7919 % 1009) as f32 - 500.0) / 7.0,
            };
            storage.update(&[value]).unwrap();
        }
        storage.flush();
        let state = &storage.states[0];
        assert_eq!(count_total(state), 2_003);
        assert_eq!(storage.state_weights[0], 2_003);
        let active = (0..RANK_KNOT_K)
            .filter(|&index| state.counts[index] != 0)
            .collect::<Vec<_>>();
        assert!(active.windows(2).all(|pair| {
            state.values[pair[0]]
                .total_cmp(&state.values[pair[1]])
                .is_le()
        }));
        for &index in &active {
            if !state.values[index].is_finite() {
                assert_ne!(state.pure_mask & (1_u64 << index), 0);
            }
        }
    }
}
