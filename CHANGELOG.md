# Changelog

Notable changes to `monatq` are documented in this file.

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

### Snapshot migration

The Rust generic default is now RankKnot. To load a TDigest snapshot created by the 0.2 default, name its kernel explicitly:

```rust
use monatq::{TDigest, TensorDigest};

let digest = TensorDigest::<f32, TDigest>::load(path)?;
```

Alternatively, use `monatq::load` or `monatq::from_bytes` to detect both the kernel and element type. Python `TensorDigest.load` and `TensorDigest.from_bytes` detect these automatically.
