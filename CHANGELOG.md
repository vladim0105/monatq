# Changelog

Notable changes to `monatq` are documented in this file.

## [Unreleased]

### Added

- **Blockwise tracking.** A digest can pool all values in each 1D group along one axis into a single shared distribution, instead of tracking every element independently. Memory scales with the number of blocks rather than the number of elements.

  ```rust
  use monatq::{BlockConfig, TensorDigest};

  // 16 blocks of 256 values per row of a [4096, 4096] weight matrix.
  let digest = TensorDigest::<f32>::with_blocks(&[4096, 4096], BlockConfig::block_size(256, -1))?;
  assert_eq!(digest.shape(), &[4096, 4096]);     // what update() accepts
  assert_eq!(digest.block_shape(), &[4096, 16]); // what queries return
  ```

  - `BlockConfig::block_size(size, axis)` makes fixed-width groups with a short final group; `BlockConfig::blocks_per_axis(count, axis)` makes balanced groups. Axes may be negative (`-1` is the last axis).
  - `TensorDigest::with_blocks` and `TensorDigest::with_block_config` construct blocked digests.
  - New accessors: `block_shape`, `block_count`, `block_axis`, `blocks_per_axis`, `block_size`. `AnyTensorDigest` gains `block_shape`.
  - Python: `TensorDigest(..., blocks=BlockConfig(block_size=... | blocks_per_axis=..., axis=...))`, the `block_size=` / `blocks_per_axis=` / `block_axis=` shorthands, and matching read-only properties including `block_shape`.
- `buffer_capacity = 0` is accepted by both kernels and compresses every sample immediately without a buffer. In 0.3.0, Rust panicked and Python raised `ValueError`.
- `TDigestConfig` gains `buffer_capacity: Option<usize>`; `None` keeps the previous `2 × compression`. Python accepts `buffer_capacity` for `kernel="tdigest"`.

### Breaking changes

- **Query results, indices, and merges are per block.** `shape()` and `numel()` keep their 0.3.0 meaning, the tensor accepted by `update`. But for a blocked digest, `quantile`, `quantiles`, `analyze`, `min`, and `max` return `block_count()` values in `block_shape()` row-major order, not `numel()` values. `cell_quantiles`, `total_weight`, `cell_min`, `cell_max`, and `merge_cells` take **block indices**, and `merge_channels` counts channels in the block grid. Code that sizes or reshapes query output from `shape()`/`numel()` must use `block_shape()`/`block_count()` for blocked digests. Elementwise digests (every existing constructor) are unaffected, since both shapes are equal.
- **The `QuantileSpine` kernel is removed**, along with `QuantileSpineConfig`, `SpineLink`, and `SpineRegime`. Use `RankKnot` (the default) or `TDigest`.
- **Snapshots from 0.3.0 and earlier can no longer be loaded.** RankKnot snapshots move to format version 6 and TDigest snapshots now begin with their own kernel tag and format version, so both record the block layout. Older snapshots are rejected with `Error::InvalidSnapshot`; regenerate them. TDigest snapshots no longer contain the ingestion row buffer, so they are smaller.
- `TDigestConfig` has a new public field, so struct literals must add it or use `..Default::default()`: `TDigestConfig { compression: 100, ..Default::default() }`.
- `buffer_capacity` now counts new values each block collects before compression, rather than buffered tensor rows. The digest buffers `ceil(buffer_capacity / block_len)` rows, so elementwise digests behave exactly as before.
- `Error::IndexOutOfBounds` now reports a block index; its message reads "block index … is out of bounds for … atomic blocks".

### Changed

- `total_weight` counts the observations pooled into a block. For elementwise digests it still equals the number of samples.
- Merges weight each source block by its observation count, so unequal-sized blocks contribute in proportion to their populations. A merged RankKnot digest continues ingesting with that combined weight; in 0.3.0 it used `sample_count × selected positions`, which is the same number for elementwise digests.
- TDigest `merge_all` merges every block in a single pass instead of going channel by channel, so results can differ slightly in the last digits from 0.3.0.

## [0.3.0]

### Breaking changes

- `RankKnot` replaces `TDigest` as the default `TensorDigest` kernel. `TDigest` remains fully supported through explicit kernel selection.
- Rust construction now separates the default constructor from kernel-specific configuration:

  ```rust
  // 0.2.2: TDigest with compression 100
  let digest = TensorDigest::<f32>::new(&[3, 4], 100);

  // 0.3.0: RankKnot with its default configuration
  let digest = TensorDigest::<f32>::new(&[3, 4]);

  // 0.3.0: explicitly retain TDigest with custom compression
  use monatq::{TDigest, TDigestConfig, TensorDigest};

  let digest = TensorDigest::<f32, TDigest>::with_config(
      &[3, 4],
      TDigestConfig { compression: 100 },
  );
  ```

  `with_config` also accepts `RankKnotConfig { buffer_capacity: ... }` for a `RankKnot` digest.

- Operations that can reject input now return `monatq::Result`: `update`, `total_weight`, `cell_quantiles`, `analyze`, `merge_cells`, `merge_channels`, `merge_all`, and `without_zeros`. Serialization, loading, and visualization now return `monatq::Result` instead of `std::io::Result`.

  ```rust
  // 0.2.2
  digest.update(&sample);
  let merged = digest.merge_all();

  // 0.3.0
  digest.update(&sample)?;
  let merged = digest.merge_all()?;
  ```

  Operations with no failure mode remain infallible: `quantile`, `quantiles`, `flush`, `numel`, and `shape`.

### Errors

- Added the public `monatq::Error` and `monatq::Result<T>` types.
- Errors are classified as `Unsupported`, `ShapeMismatch`, `IndexOutOfBounds`, `InvalidConfig`, `InvalidSnapshot`, or `Io`, replacing panics and undifferentiated I/O errors where applicable.

### Distribution analysis

- Distribution classification is now shared across kernels, so RankKnot and TDigest expose the same classifications.
- `Distribution::BiNormal` identifies a symmetric mixture of two Gaussian components. This variant also existed in 0.2.2; exhaustive matches must continue to handle it along with `Unknown`.

### Python

- `TensorDigest` now defaults to `kernel="rankknot"`. Select the previous backend explicitly with `kernel="tdigest"`.
- Arguments after `shape` are keyword-only.

  ```python
  # 0.2.2
  digest = TensorDigest([3, 4], 100)

  # 0.3.0: default RankKnot
  digest = TensorDigest([3, 4])

  # 0.3.0: explicitly retain TDigest
  digest = TensorDigest(
      [3, 4],
      kernel="tdigest",
      compression=100,
  )

  # Tune RankKnot
  digest = TensorDigest([3, 4], buffer_capacity=512)
  ```

- `compression` is accepted only with `kernel="tdigest"`; `buffer_capacity` is accepted only with `kernel="rankknot"`. Invalid combinations raise `ValueError`.
- Added the read-only `kernel` property.
- Rust errors are translated to catchable Python exceptions, including `ValueError`, `IndexError`, `NotImplementedError`, and `IOError`.

### Snapshots

The Rust generic default is now RankKnot. To load a current-format TDigest snapshot with the typed loader, name its kernel explicitly:

```rust
use monatq::{TDigest, TensorDigest};

let digest = TensorDigest::<f32, TDigest>::load(path)?;
```

Alternatively, use `monatq::load` or `monatq::from_bytes` to detect both the kernel and element type. Python `TensorDigest.load` and `TensorDigest.from_bytes` detect these automatically.
