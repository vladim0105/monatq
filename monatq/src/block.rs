use crate::{Error, Result};

/// How a digest groups tensor elements into statistical blocks.
///
/// Blocks are one-dimensional groups along a single axis, never 2D tiles. Negative axes
/// count from the end (`-1` is the last axis) and are resolved against the shape when a
/// digest is constructed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum BlockConfig {
    /// Track every element independently.
    #[default]
    Elementwise,
    /// Fixed-width groups of `size` values along `axis`, with a short final group when the
    /// axis length is not a multiple of `size`. `size` must be positive.
    Size { size: usize, axis: isize },
    /// `count` balanced groups along `axis`, larger groups first. `count` must be positive
    /// and is clamped to the axis length.
    Count { count: usize, axis: isize },
}

impl BlockConfig {
    /// Track every element independently. This is the default.
    pub const fn elementwise() -> Self {
        Self::Elementwise
    }

    /// Fixed-width groups with a short final group. Size must be positive.
    pub const fn block_size(size: usize, axis: isize) -> Self {
        Self::Size { size, axis }
    }

    /// Balanced groups. Count must be positive and is clamped to the axis length.
    pub const fn blocks_per_axis(count: usize, axis: isize) -> Self {
        Self::Count { count, axis }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
enum Grouping {
    Size(usize),
    /// Stores the effective count, already clamped to the axis length.
    Count(usize),
    Elementwise,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct BlockLayout {
    input_shape: Vec<usize>,
    shape: Vec<usize>,
    input_numel: usize,
    block_count: usize,
    grouping: Grouping,
    axis: usize,
    inner: usize,
    axis_len: usize,
    blocks_axis: usize,
    base_block_len: usize,
    larger_blocks: usize,
}

impl BlockLayout {
    pub(crate) fn new(shape: &[usize], config: BlockConfig) -> Result<Self> {
        let (grouping, requested_axis) = match config {
            BlockConfig::Elementwise if shape.is_empty() => return Ok(Self::scalar()),
            BlockConfig::Elementwise => (Grouping::Elementwise, -1),
            BlockConfig::Size { size: 0, .. } => {
                return Err(Error::InvalidConfig {
                    parameter: "block size",
                    message: "must be positive",
                });
            }
            BlockConfig::Size { size, axis } => (Grouping::Size(size), axis),
            BlockConfig::Count { count: 0, .. } => {
                return Err(Error::InvalidConfig {
                    parameter: "blocks per axis",
                    message: "must be positive; use BlockConfig::Elementwise for elementwise tracking",
                });
            }
            BlockConfig::Count { count, axis } => (Grouping::Count(count), axis),
        };
        let axis = if requested_axis < 0 {
            shape.len().checked_sub(requested_axis.unsigned_abs())
        } else {
            Some(requested_axis as usize)
        }
        .filter(|&axis| axis < shape.len())
        .ok_or(Error::InvalidConfig {
            parameter: "block axis",
            message: "must name an existing tensor axis",
        })?;
        let numel = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or(Error::InvalidConfig {
                parameter: "shape",
                message: "element count overflows usize",
            })?;
        let axis_len = shape[axis];
        let grouping = match grouping {
            Grouping::Count(count) => Grouping::Count(count.min(axis_len).max(1)),
            other => other,
        };
        let blocks_axis = match grouping {
            Grouping::Elementwise => axis_len,
            Grouping::Size(size) => axis_len.div_ceil(size),
            Grouping::Count(count) => count.min(axis_len),
        };
        let (base_block_len, larger_blocks) = if blocks_axis == 0 {
            (0, 0)
        } else {
            (axis_len / blocks_axis, axis_len % blocks_axis)
        };
        let product = |dims: &[usize]| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(Error::InvalidConfig {
                    parameter: "shape",
                    message: "axis stride overflows usize",
                })
        };
        let inner = product(&shape[axis + 1..])?;
        let outer = product(&shape[..axis])?;
        let block_count = outer
            .checked_mul(blocks_axis)
            .and_then(|v| v.checked_mul(inner))
            .ok_or(Error::InvalidConfig {
                parameter: "block layout",
                message: "block count overflows usize",
            })?;
        let mut compact_shape = shape.to_vec();
        compact_shape[axis] = blocks_axis;
        Ok(Self {
            input_shape: shape.to_vec(),
            shape: compact_shape,
            input_numel: numel,
            block_count,
            grouping,
            axis,
            inner,
            axis_len,
            blocks_axis,
            base_block_len,
            larger_blocks,
        })
    }

    /// A scalar (empty-shape) digest has no axis to index, so it gets fixed single-block
    /// metadata that the span arithmetic can use safely.
    fn scalar() -> Self {
        Self {
            input_shape: vec![],
            shape: vec![],
            input_numel: 1,
            block_count: 1,
            grouping: Grouping::Elementwise,
            axis: 0,
            inner: 1,
            axis_len: 1,
            blocks_axis: 1,
            base_block_len: 1,
            larger_blocks: 0,
        }
    }

    pub(crate) fn default_for(shape: &[usize]) -> Self {
        Self::new(shape, BlockConfig::Elementwise).expect("elementwise block layout")
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let rebuilt = Self::new(&self.input_shape, self.config())
            .map_err(|error| Error::InvalidSnapshot(error.to_string()))?;
        if &rebuilt != self {
            return Err(Error::InvalidSnapshot(
                "inconsistent block layout metadata".into(),
            ));
        }
        Ok(())
    }

    /// The resolved configuration: a nonnegative axis and, in count mode, the effective count.
    pub(crate) fn config(&self) -> BlockConfig {
        let axis = self.axis as isize;
        match self.grouping {
            Grouping::Elementwise => BlockConfig::Elementwise,
            Grouping::Size(size) => BlockConfig::Size { size, axis },
            Grouping::Count(count) => BlockConfig::Count { count, axis },
        }
    }

    /// Flat input positions of `block` as `(first, stride, len)`.
    #[inline]
    pub(crate) fn span(&self, block: usize) -> (usize, usize, usize) {
        let inner_pos = block % self.inner;
        let t = block / self.inner;
        let block_axis = if self.blocks_axis == 0 {
            0
        } else {
            t % self.blocks_axis
        };
        let outer = if self.blocks_axis == 0 {
            0
        } else {
            t / self.blocks_axis
        };
        let (start_axis, len) = self.axis_range(block_axis);
        (
            (outer * self.axis_len + start_axis) * self.inner + inner_pos,
            self.inner,
            len,
        )
    }

    /// Length of the largest block; remainder blocks may be shorter.
    pub(crate) fn max_block_len(&self) -> usize {
        let len = match self.grouping {
            Grouping::Size(size) => size.min(self.axis_len),
            Grouping::Count(_) | Grouping::Elementwise => {
                self.base_block_len + usize::from(self.larger_blocks > 0)
            }
        };
        len.max(1)
    }

    /// Tensor rows to buffer so each full-size block collects `values_per_block` new values
    /// before compression. Zero means process every sample directly: buffering a single row
    /// would only add a copy.
    pub(crate) fn buffer_rows(&self, values_per_block: usize) -> usize {
        match values_per_block.div_ceil(self.max_block_len()) {
            0 | 1 => 0,
            rows => rows,
        }
    }

    fn axis_range(&self, block_axis: usize) -> (usize, usize) {
        match self.grouping {
            Grouping::Size(size) => {
                let start = block_axis * size;
                (start, size.min(self.axis_len - start))
            }
            Grouping::Count(_) | Grouping::Elementwise => (
                block_axis * self.base_block_len + block_axis.min(self.larger_blocks),
                self.base_block_len + usize::from(block_axis < self.larger_blocks),
            ),
        }
    }

    pub(crate) fn input_numel(&self) -> usize {
        self.input_numel
    }
    pub(crate) fn block_count(&self) -> usize {
        self.block_count
    }
    pub(crate) fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Layout that reinterprets this digest as `shape` grouped by `config`.
    ///
    /// `new_to_old` is `None` when every block keeps its compact index. A `Some` map sends
    /// each new block index to the old block that covers the same elements, so callers can
    /// move summaries without touching the flat element buffer.
    pub(crate) fn retarget(
        &self,
        shape: &[usize],
        config: BlockConfig,
        operation: &'static str,
    ) -> Result<LayoutChange> {
        let layout = Self::new(shape, config)?;
        let new_to_old = self.block_order(&layout, operation)?;
        Ok(LayoutChange { layout, new_to_old })
    }

    fn block_order(
        &self,
        target: &Self,
        operation: &'static str,
    ) -> Result<Option<Vec<usize>>> {
        if self.input_numel != target.input_numel {
            return Err(incompatible(
                operation,
                "element count must be unchanged",
            ));
        }
        // Elementwise (and any other one-element grouping) is a pure shape change: block i
        // is flat element i on both sides, so summaries and the row buffer already agree.
        if self.each_element_is_a_block() && target.each_element_is_a_block() {
            return Ok(None);
        }
        if self.block_count == target.block_count && self.spans_equal(target) {
            return Ok(None);
        }
        if target.block_count > self.block_count {
            return Err(incompatible(operation, "it would split a pooled block"));
        }
        if target.block_count < self.block_count {
            return Err(incompatible(operation, "it would merge distinct blocks"));
        }
        self.permuted_order(target, operation)
    }

    fn each_element_is_a_block(&self) -> bool {
        self.block_count == self.input_numel && self.singleton_blocks()
    }

    fn singleton_blocks(&self) -> bool {
        match self.grouping {
            Grouping::Elementwise => true,
            Grouping::Size(size) => size == 1 || self.axis_len <= 1,
            Grouping::Count(_) => self.larger_blocks == 0 && self.base_block_len <= 1,
        }
    }

    fn spans_equal(&self, target: &Self) -> bool {
        (0..self.block_count).all(|block| self.span(block) == target.span(block))
    }

    /// Equal-sized partitions whose compact order may differ.
    fn permuted_order(&self, target: &Self, operation: &'static str) -> Result<Option<Vec<usize>>> {
        let mut new_to_old = vec![usize::MAX; target.block_count];
        let mut identity = true;
        for old_block in 0..self.block_count {
            let (start, stride, len) = self.span(old_block);
            let new_block = target.block_index(start);
            for step in 1..len {
                if target.block_index(start + step * stride) != new_block {
                    return Err(incompatible(operation, "it would split a pooled block"));
                }
            }
            let new_span = target.span(new_block);
            if new_span != (start, stride, len) {
                let reason = if new_span.2 > len {
                    "it would merge distinct blocks"
                } else {
                    "it would split a pooled block"
                };
                return Err(incompatible(operation, reason));
            }
            if new_to_old[new_block] != usize::MAX {
                return Err(incompatible(operation, "it would merge distinct blocks"));
            }
            new_to_old[new_block] = old_block;
            identity &= new_block == old_block;
        }
        if new_to_old.iter().any(|&old| old == usize::MAX) {
            return Err(incompatible(operation, "it would split a pooled block"));
        }
        Ok((!identity).then_some(new_to_old))
    }

    /// Compact block index that owns flat input position `flat`.
    fn block_index(&self, flat: usize) -> usize {
        debug_assert!(self.inner > 0 && self.axis_len > 0);
        let inner_pos = flat % self.inner;
        let along = flat / self.inner;
        let axis_pos = along % self.axis_len;
        let outer = along / self.axis_len;
        let block_axis = self.axis_block(axis_pos);
        (outer * self.blocks_axis + block_axis) * self.inner + inner_pos
    }

    fn axis_block(&self, axis_pos: usize) -> usize {
        match self.grouping {
            Grouping::Size(size) => axis_pos / size,
            Grouping::Count(_) | Grouping::Elementwise => {
                let large = self.base_block_len + 1;
                let head = self.larger_blocks * large;
                if self.larger_blocks > 0 && axis_pos < head {
                    axis_pos / large
                } else if self.base_block_len == 0 {
                    0
                } else {
                    self.larger_blocks + (axis_pos - head) / self.base_block_len
                }
            }
        }
    }
}

/// A checked replacement for a digest's [`BlockLayout`].
#[derive(Debug)]
pub(crate) struct LayoutChange {
    pub layout: BlockLayout,
    /// `new_to_old[new_block] = old_block`. `None` means the compact order is unchanged.
    pub new_to_old: Option<Vec<usize>>,
}

fn incompatible(operation: &'static str, reason: &'static str) -> Error {
    Error::IncompatibleLayout { operation, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every input position belongs to exactly one block, and each block stays within the
    /// outer and inner coordinates its compact index names.
    fn assert_partition(layout: &BlockLayout) {
        let mut visits = vec![0; layout.input_numel()];
        for block in 0..layout.block_count() {
            let (start, stride, len) = layout.span(block);
            assert_eq!(stride, layout.inner);
            assert_eq!(start % layout.inner, block % layout.inner);
            assert_eq!(
                start / (layout.inner * layout.axis_len),
                block / (layout.inner * layout.blocks_axis)
            );
            for k in 0..len {
                let flat = start + k * stride;
                visits[flat] += 1;
                assert_eq!(layout.block_index(flat), block);
            }
        }
        assert!(visits.iter().all(|&count| count == 1));
    }

    #[test]
    fn largest_axis_can_map_to_one_block_without_overflow() {
        let layout = BlockLayout::new(&[usize::MAX], BlockConfig::blocks_per_axis(1, 0)).unwrap();
        assert_eq!(layout.span(0), (0, 1, usize::MAX));
    }

    #[test]
    fn buffer_rows_fill_the_largest_block() {
        let rows = |shape: &[usize], config, capacity| {
            BlockLayout::new(shape, config)
                .unwrap()
                .buffer_rows(capacity)
        };
        assert_eq!(rows(&[4, 16], BlockConfig::default(), 256), 256);
        assert_eq!(rows(&[4, 16], BlockConfig::block_size(8, 1), 256), 32);
        assert_eq!(rows(&[4, 16], BlockConfig::block_size(3, 1), 7), 3);
        assert_eq!(rows(&[4, 16], BlockConfig::block_size(8, 1), 8), 0);
        assert_eq!(rows(&[4, 16], BlockConfig::block_size(64, 1), 16), 0);
        assert_eq!(rows(&[4, 16], BlockConfig::block_size(64, 1), 32), 2);
        assert_eq!(rows(&[4, 17], BlockConfig::blocks_per_axis(4, 1), 10), 2);
        assert_eq!(rows(&[4, 16], BlockConfig::default(), 0), 0);
        assert_eq!(rows(&[4, 0], BlockConfig::default(), 4), 4);
    }

    #[test]
    fn fixed_size_mapping_covers_every_position_once() {
        for length in 0..40 {
            for size in 1..45 {
                assert_partition(
                    &BlockLayout::new(&[2, length, 3], BlockConfig::block_size(size, 1)).unwrap(),
                );
            }
        }
        let layout =
            BlockLayout::new(&[usize::MAX], BlockConfig::block_size(usize::MAX - 1, 0)).unwrap();
        assert_eq!(layout.span(1), (usize::MAX - 1, 1, 1));
    }

    #[test]
    fn balanced_mapping_covers_every_position_once() {
        for length in 0..40 {
            for requested in 1..45 {
                assert_partition(
                    &BlockLayout::new(&[2, length, 3], BlockConfig::blocks_per_axis(requested, 1))
                        .unwrap(),
                );
            }
            assert_partition(&BlockLayout::new(&[2, length, 3], BlockConfig::Elementwise).unwrap());
        }
    }

    #[test]
    fn zero_is_rejected_in_both_modes() {
        for config in [
            BlockConfig::block_size(0, 1),
            BlockConfig::blocks_per_axis(0, 1),
        ] {
            assert!(matches!(
                BlockLayout::new(&[2, 5], config),
                Err(Error::InvalidConfig { .. })
            ));
        }
    }

    #[test]
    fn config_reports_resolved_axis_and_effective_count() {
        let config = |shape: &[usize], config| BlockLayout::new(shape, config).unwrap().config();
        assert_eq!(
            config(&[2, 5], BlockConfig::blocks_per_axis(99, -1)),
            BlockConfig::Count { count: 5, axis: 1 }
        );
        assert_eq!(
            config(&[2, 5], BlockConfig::block_size(99, -2)),
            BlockConfig::Size { size: 99, axis: 0 }
        );
        assert_eq!(
            config(&[2, 0], BlockConfig::blocks_per_axis(4, 1)),
            BlockConfig::Count { count: 1, axis: 1 }
        );
        assert_eq!(
            config(&[2, 5], BlockConfig::Elementwise),
            BlockConfig::Elementwise
        );
        assert_eq!(
            config(&[], BlockConfig::Elementwise),
            BlockConfig::Elementwise
        );
    }

    #[test]
    fn elementwise_blocks_are_the_input_positions() {
        let layout = BlockLayout::new(&[2, 3, 4], BlockConfig::Elementwise).unwrap();
        assert_eq!(layout.shape(), &[2, 3, 4]);
        for block in 0..layout.block_count() {
            assert_eq!(layout.span(block), (block, 1, 1));
        }
    }

    #[test]
    fn elementwise_retarget_keeps_block_order() {
        let layout = BlockLayout::new(&[2, 3], BlockConfig::Elementwise).unwrap();
        let change = layout
            .retarget(&[3, 2], BlockConfig::Elementwise, "reshape")
            .unwrap();
        assert!(change.new_to_old.is_none());
        assert_eq!(change.layout.input_shape(), &[3, 2]);
        assert_eq!(change.layout.block_count(), 6);

        let scalar = BlockLayout::new(&[], BlockConfig::Elementwise).unwrap();
        let widened = scalar
            .retarget(&[1, 1], BlockConfig::Elementwise, "reshape")
            .unwrap();
        assert!(widened.new_to_old.is_none());
        assert_eq!(widened.layout.input_numel(), 1);

        let empty = BlockLayout::new(&[0, 4], BlockConfig::Elementwise).unwrap();
        let still_empty = empty
            .retarget(&[2, 0, 3], BlockConfig::Elementwise, "reshape")
            .unwrap();
        assert!(still_empty.new_to_old.is_none());
        assert_eq!(still_empty.layout.block_count(), 0);
    }

    #[test]
    fn compatible_blocked_retarget_preserves_spans() {
        let layout = BlockLayout::new(&[2, 8], BlockConfig::block_size(4, -1)).unwrap();
        let change = layout
            .retarget(&[4, 4], BlockConfig::block_size(4, 1), "reshape")
            .unwrap();
        assert!(change.new_to_old.is_none());
        assert!(layout.spans_equal(&change.layout));

        let flat = layout
            .retarget(&[16], BlockConfig::block_size(4, 0), "remap")
            .unwrap();
        assert!(flat.new_to_old.is_none());
        assert!(layout.spans_equal(&flat.layout));
    }

    #[test]
    fn retarget_rejects_split_merge_and_element_count_changes() {
        let pooled = BlockLayout::new(&[2, 8], BlockConfig::block_size(4, -1)).unwrap();
        let split = pooled
            .retarget(&[8, 2], BlockConfig::block_size(4, 1), "reshape")
            .unwrap_err();
        assert_eq!(
            split.to_string(),
            "cannot reshape this digest: it would split a pooled block"
        );

        let crossed = pooled
            .retarget(&[2, 8], BlockConfig::block_size(2, 0), "remap")
            .unwrap_err();
        assert!(crossed.to_string().contains("split a pooled block"));

        // Same block count, but the pools run along the other axis.
        let rows = BlockLayout::new(&[2, 4], BlockConfig::block_size(2, 0)).unwrap();
        let columns = rows
            .retarget(&[2, 4], BlockConfig::block_size(2, 1), "remap")
            .unwrap_err();
        assert!(columns.to_string().contains("split a pooled block"));

        let elementwise = BlockLayout::new(&[8], BlockConfig::Elementwise).unwrap();
        let merge = elementwise
            .retarget(&[8], BlockConfig::block_size(4, 0), "remap")
            .unwrap_err();
        assert_eq!(
            merge.to_string(),
            "cannot remap this digest: it would merge distinct blocks"
        );

        let count = elementwise
            .retarget(&[4], BlockConfig::Elementwise, "reshape")
            .unwrap_err();
        assert_eq!(
            count.to_string(),
            "cannot reshape this digest: element count must be unchanged"
        );
    }

    #[test]
    fn block_index_round_trips_the_short_final_group() {
        let layout = BlockLayout::new(&[1, 5], BlockConfig::block_size(2, 1)).unwrap();
        assert_partition(&layout);
        let balanced = BlockLayout::new(&[1, 5], BlockConfig::blocks_per_axis(2, 1)).unwrap();
        assert_partition(&balanced);
        let huge = BlockLayout::new(&[usize::MAX], BlockConfig::block_size(usize::MAX - 1, 0)).unwrap();
        assert_eq!(huge.block_index(0), 0);
        assert_eq!(huge.block_index(usize::MAX - 1), 1);
    }
}
