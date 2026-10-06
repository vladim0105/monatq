# Changelog

Notable changes to `monatq` are documented in this file.

## [Unreleased]

### Added

- **Reshape and remap.** `TensorDigest::reshape` changes the shape accepted by `update` without flushing or recompressing. `remap` changes that shape and the block grouping together. Elementwise digests accept any shape with the same element count. A blocked digest moves only when every block still covers the same elements. Splitting a pooled block, or merging blocks that are still separate, returns `Error::IncompatibleLayout` and leaves the digest unchanged. Pending rows stay in flat element order. Python: `digest.reshape(shape)` and `digest.remap(shape, blocks=None)`, both raising `ValueError`.
- **Permute.** `TensorDigest::permute(axes)` reorders the tensor axes like numpy's `transpose(axes)` or torch's `permute`. Unlike `reshape` and `remap`, it moves elements: every element's summary and any rows still waiting to be compressed move with it, and nothing is flushed or recompressed. A block grouping follows its axis, so blocked digests are never rejected. `axes` must name every axis once (negative axes allowed); otherwise `Error::InvalidConfig` is returned and the digest is unchanged. Python: `digest.permute(axes)`, raising `ValueError`.

## [0.4.1]

### Changed

- **RankKnot can merge tensors of a million elements or more.** 0.4.0 reserved 32 knots × 16 bytes of scratch per selected block before sorting: **512 MB at 1 million elements, 5.12 GB at 10 million**. `merge_cells`, `merge_channels`, and `merge_all` now reuse one **32 KiB** buffer (15,625× less scratch at 1 million elements). More than 64 blocks go through a 64-way tree, four recompressions at 1 million elements. Extrema and population totals stay exact; quantiles on those wider merges can differ from 0.4.0.

## [0.4.0]

### Added

- **Blockwise tracking.** A digest can pool all values in each 1D group along one axis into a single shared distribution, instead of tracking every element independently. Memory scales with the number of blocks rather than the number of elements.

  ```rust
  use monatq::{BlockConfig, TensorDigest};

  // 16 blocks of 256 values per row of a [4096, 4096] weight matrix.
  let digest = TensorDigest::<f32>::with_blocks(&[4096, 4096], BlockConfig::block_size(256, -1))?;
  assert_eq!(digest.shape(), &[4096, 4096]);     // what update() accepts
  assert_eq!(digest.block_shape(), &[4096, 16]); // what queries return
  ```

  - `BlockConfig` is an enum: `Elementwise` (the default), `Size { size, axis }` for fixed-width groups with a short final group, and `Count { count, axis }` for balanced groups clamped to the axis length. The `BlockConfig::elementwise()`, `BlockConfig::block_size(size, axis)` and `BlockConfig::blocks_per_axis(count, axis)` constructors build them. Size and count must be positive. Axes may be negative (`-1` is the last axis).
  - `TensorDigest::with_blocks` and `TensorDigest::with_block_config` construct blocked digests.
  - New accessors: `block_shape`, `block_count`, and `block_config`. `block_config` returns the resolved grouping, with a nonnegative axis and, in count mode, the effective count. `AnyTensorDigest` gains `block_shape`.
  - Python: `TensorDigest(..., blocks=BlockConfig(block_size=... | blocks_per_axis=..., axis=...))`, the `block_size=` / `blocks_per_axis=` shorthands with optional `block_axis=`, and the read-only `block_shape`, `block_count` and `block_config` properties. `block_config` is `None` for elementwise digests. `BlockConfig` supports `==` and has a readable `repr`.
- `buffer_capacity = 0` is accepted by both kernels and compresses every sample immediately without a buffer. In 0.3.0, Rust panicked and Python raised `ValueError`.
- `TDigestConfig` gains `buffer_capacity: Option<usize>`; `None` keeps the previous `2 × compression`. Python accepts `buffer_capacity` for `kernel="tdigest"`.

### Breaking changes

- **Query results, indices, and merges are per block.** `shape()` and `numel()` keep their 0.3.0 meaning, the tensor accepted by `update`. But for a blocked digest, `quantile`, `quantiles`, `analyze`, `min`, and `max` return `block_count()` values in `block_shape()` row-major order, not `numel()` values. `cell_quantiles`, `total_weight`, `cell_min`, `cell_max`, and `merge_cells` take **block indices**, and `merge_channels` counts channels in the block grid. Code that sizes or reshapes query output from `shape()`/`numel()` must use `block_shape()`/`block_count()` for blocked digests. Elementwise digests (every existing constructor) are unaffected, since both shapes are equal.
- **The `QuantileSpine` kernel is removed**, along with `QuantileSpineConfig`, `SpineLink`, and `SpineRegime`. Use `RankKnot` (the default) or `TDigest`.
- **Snapshots from 0.3.0 and earlier can no longer be loaded.** RankKnot snapshots move to format version 7 and TDigest snapshots now begin with their own kernel tag and format version, so both record the block layout. Older snapshots are rejected with `Error::InvalidSnapshot`; regenerate them. TDigest snapshots no longer contain the ingestion row buffer, so they are smaller.
- `TDigestConfig` has a new public field, so struct literals must add it or use `..Default::default()`: `TDigestConfig { compression: 100, ..Default::default() }`.
- `buffer_capacity` now counts new values each block collects before compression, rather than buffered tensor rows. The digest buffers `ceil(buffer_capacity / block_len)` rows, so elementwise digests behave exactly as before.
- `Error::IndexOutOfBounds` now reports a block index; its message reads "block index … is out of bounds for … atomic blocks".

### Changed

- **RankKnot stores exact observation counts per knot** instead of 16-bit probability masses normalized to 65,535. Rounding onto that grid discarded any update smaller than half a step, so with small buffers or long streams the summary partly stopped learning: at 10 million samples, 0.3.0's default had about 8× its 100,000-sample error, and `buffer_capacity: 0` reached about 6% mean rank error. Merging is now exact integer addition, and accuracy no longer depends on `buffer_capacity` or stream length. Per-block state grows from 208 to 272 bytes, and updates are about 18% faster at the same buffer size.
- RankKnot's default `buffer_capacity` drops from 256 to 16. With exact counts this costs no accuracy. Each block uses about 344 bytes instead of about 1,240, so elementwise digests use about 3.6× less memory. Compressing more often slows updates; pass `RankKnotConfig { buffer_capacity: 256 }` for more throughput. `buffer_capacity` is not stored in snapshots, so loaded digests also use the new default.
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
