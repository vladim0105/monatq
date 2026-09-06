use crate::{Error, Result};

/// One-dimensional statistical groups along a tensor axis, never 2D tiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlockConfig {
    grouping: Grouping,
    /// Axis to partition. Negative indices count from the end (`-1` is the last axis).
    /// Resolved and validated against the shape when constructing a digest.
    pub axis: isize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
enum Grouping {
    Size(usize),
    Count(usize),
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self::blocks_per_axis(0, 0)
    }
}

impl BlockConfig {
    /// Fixed-width groups with a short final group. Size must be positive.
    pub const fn block_size(size: usize, axis: isize) -> Self {
        Self {
            grouping: Grouping::Size(size),
            axis,
        }
    }

    /// Balanced groups. Zero means elementwise; counts are clamped to the axis length.
    pub const fn blocks_per_axis(count: usize, axis: isize) -> Self {
        Self {
            grouping: Grouping::Count(count),
            axis,
        }
    }

    /// Compatibility alias for [`Self::blocks_per_axis`].
    pub const fn new(blocks_per_axis: usize, axis: isize) -> Self {
        Self::blocks_per_axis(blocks_per_axis, axis)
    }
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
        let axis = if config.axis < 0 {
            shape.len().checked_sub(config.axis.unsigned_abs())
        } else {
            Some(config.axis as usize)
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
        let blocks_axis = match config.grouping {
            Grouping::Size(0) => {
                return Err(Error::InvalidConfig {
                    parameter: "block size",
                    message: "must be positive",
                });
            }
            Grouping::Size(size) => axis_len.div_ceil(size),
            Grouping::Count(0) => axis_len,
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
            grouping: config.grouping,
            axis,
            inner,
            axis_len,
            blocks_axis,
            base_block_len,
            larger_blocks,
        })
    }

    pub(crate) fn default_for(shape: &[usize]) -> Self {
        Self::try_default_for(shape).expect("default block layout")
    }

    pub(crate) fn try_default_for(shape: &[usize]) -> Result<Self> {
        // Existing constructors accept scalar/empty shapes. Preserve that public geometry while
        // using safe scalar metadata internally.
        if shape.is_empty() {
            Ok(Self {
                input_shape: vec![],
                shape: vec![],
                input_numel: 1,
                block_count: 1,
                grouping: Grouping::Count(0),
                axis: 0,
                inner: 1,
                axis_len: 1,
                blocks_axis: 1,
                base_block_len: 1,
                larger_blocks: 0,
            })
        } else {
            Self::new(shape, BlockConfig::default())
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let rebuilt = if self.input_shape.is_empty() {
            Self::try_default_for(&self.input_shape)
        } else {
            Self::new(
                &self.input_shape,
                BlockConfig {
                    grouping: self.grouping,
                    axis: isize::try_from(self.axis)
                        .map_err(|_| Error::InvalidSnapshot("block axis overflows isize".into()))?,
                },
            )
        }
        .map_err(|error| Error::InvalidSnapshot(error.to_string()))?;
        if &rebuilt != self {
            return Err(Error::InvalidSnapshot(
                "inconsistent block layout metadata".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn block_of(&self, flat: usize) -> usize {
        if self.is_elementwise() {
            return flat;
        }
        let inner_pos = flat % self.inner;
        let axis_coord = (flat / self.inner) % self.axis_len;
        let outer = flat / (self.inner * self.axis_len);
        let threshold = self.larger_blocks * self.base_block_len + self.larger_blocks;
        let block_axis = if let Grouping::Size(size) = self.grouping {
            axis_coord / size
        } else if axis_coord < threshold {
            axis_coord / (self.base_block_len + 1)
        } else {
            self.larger_blocks + (axis_coord - threshold) / self.base_block_len
        };
        (outer * self.blocks_axis + block_axis) * self.inner + inner_pos
    }

    pub(crate) fn indices(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
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
        (start_axis..start_axis + len)
            .map(move |a| (outer * self.axis_len + a) * self.inner + inner_pos)
    }

    fn axis_range(&self, block_axis: usize) -> (usize, usize) {
        match self.grouping {
            Grouping::Size(size) => {
                let start = block_axis * size;
                (start, size.min(self.axis_len - start))
            }
            Grouping::Count(_) => (
                block_axis * self.base_block_len + block_axis.min(self.larger_blocks),
                self.base_block_len + usize::from(block_axis < self.larger_blocks),
            ),
        }
    }

    pub(crate) fn block_size(&self) -> Option<usize> {
        match self.grouping {
            Grouping::Size(size) => Some(size),
            Grouping::Count(_) => None,
        }
    }

    pub(crate) fn is_elementwise(&self) -> bool {
        self.blocks_axis == self.axis_len
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
    pub(crate) fn axis(&self) -> usize {
        self.axis
    }
    pub(crate) fn blocks_per_axis(&self) -> usize {
        match self.grouping {
            Grouping::Count(count) => count,
            Grouping::Size(_) => self.blocks_axis,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_axis_can_map_to_one_block_without_overflow() {
        let layout = BlockLayout::new(&[usize::MAX], BlockConfig::new(1, 0)).unwrap();
        assert_eq!(layout.block_of(0), 0);
        assert_eq!(layout.block_of(usize::MAX - 1), 0);
    }

    #[test]
    fn fixed_size_mapping_covers_every_position_once() {
        for length in 0..40 {
            for size in 1..45 {
                let layout =
                    BlockLayout::new(&[2, length, 3], BlockConfig::block_size(size, 1)).unwrap();
                let mut visits = vec![0; layout.input_numel()];
                for block in 0..layout.block_count() {
                    for element in layout.indices(block) {
                        assert_eq!(layout.block_of(element), block);
                        visits[element] += 1;
                    }
                }
                assert!(visits.iter().all(|&count| count == 1));
            }
        }
        let layout =
            BlockLayout::new(&[usize::MAX], BlockConfig::block_size(usize::MAX - 1, 0)).unwrap();
        assert_eq!(layout.block_of(usize::MAX - 1), 1);
        assert_eq!(layout.indices(1).collect::<Vec<_>>(), vec![usize::MAX - 1]);
    }

    #[test]
    fn balanced_mapping_covers_every_position_once() {
        for length in 1..40 {
            for requested in 0..45 {
                let layout =
                    BlockLayout::new(&[2, length, 3], BlockConfig::new(requested, 1)).unwrap();
                let mut visits = vec![0; layout.input_numel()];
                for block in 0..layout.block_count() {
                    for element in layout.indices(block) {
                        assert_eq!(layout.block_of(element), block);
                        visits[element] += 1;
                    }
                }
                assert!(visits.iter().all(|&count| count == 1));
            }
        }
    }
}
