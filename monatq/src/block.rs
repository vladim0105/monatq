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
                visits[start + k * stride] += 1;
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
}
