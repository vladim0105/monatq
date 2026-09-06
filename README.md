# monatq

**Monakhov Tensor Quantiles** - approximate quantile tracking for tensors.

`monatq` provides a unified `TensorDigest<T, K>` container with statically selected `RankKnot` and `TDigest` kernels. `K` defaults to `RankKnot`. Each kernel retains its optimized flat storage layout, and updates are parallelised element-wise via Rayon.

## Use Cases

- **Model interpretability** - feed activation or weight tensors through a forward pass and query per-position quantiles to understand how individual neurons or channels behave across inputs.
- **Quantization & pruning** - characterise the value distribution at each position to inform bit-width selection, clipping ranges, or sparsity thresholds without storing all observations in memory.

![monatq visualizer](https://raw.githubusercontent.com/vladim0105/monatq/master/example.png)

## Python

See [monatq-py/README.md](https://github.com/vladim0105/monatq/blob/master/monatq-py/README.md) for the Python bindings, including installation and usage with NumPy and PyTorch.

## Rust

```bash
cargo add monatq
```

### Usage

```rust
use monatq::TensorDigest;

// Track a [3, 4] tensor (12 elements) with the default kernel (RankKnot).
let mut digest = TensorDigest::<f32>::new(&[3, 4]);

// Feed samples (row-major flat slices). `update` rejects a sample whose length
// does not match the tensor, leaving the digest untouched.
for sample in my_tensor_samples {
    digest.update(&sample)?;
}

// Query the per-element median
let medians: Vec<f32> = digest.quantile(0.5);

// Query multiple quantiles at once
let [p10, p50, p90] = digest.quantiles(&[0.1, 0.5, 0.9])[..] else { panic!() };

// Classify the distribution shape at each position
let distributions = digest.analyze()?;
```

### Blockwise tracking for quantization

Blockwise tracking pools all values in each group into one shared distribution instead of
keeping independent statistics for every tensor element. Blocks are **1D groups along one
selected axis**, independently for each combination of the other coordinates—not square
or 2D tiles. Groups never cross the other axes.

Choose the grouping explicitly with `BlockConfig`:

- `BlockConfig::block_size(size, axis)`: fixed-width groups, matching common blockwise
  quantization geometry. Size must be positive. A short final group holds the remainder:
  129 values with size 8 gives 16 groups of 8 and one of 1. No padding enters statistics.
- `BlockConfig::blocks_per_axis(count, axis)`: balanced groups. Splitting 129 values into
  16 blocks gives one of 9 and 15 of 8. **0 means elementwise**, **1 pools the entire axis**,
  and counts are clamped to the axis length. Larger groups come first.

`BlockConfig::new(count, axis)` remains an alias for balanced count-based grouping.
Both Rust and Python accept signed axes: `-1` is the last axis, `-2` the penultimate.
Axes are resolved against the input shape; out-of-range axes are rejected.
`block_axis()` (Python: `block_axis`) reports the resolved nonnegative index.

For weights shaped `[out_features, in_features]`, grouping along the last axis keeps each
output channel independent. Values are **not averaged** before ingestion: outliers and
repeated values contribute to the shared distribution. Tracking memory scales with the
number of groups rather than the number of original elements; the input tensor itself is
unchanged. Quantile results are compact, with one result per group rather than a broadcast
copy for every input element.

```rust
use monatq::{BlockConfig, TensorDigest};

// 16 blocks per output row, each pooling 256 weights.
let mut digest = TensorDigest::<f32>::with_blocks(
    &[4096, 4096], BlockConfig::block_size(256, -1),
)?;
assert_eq!(digest.shape(), &[4096, 16]);
// update() still accepts the complete original tensor.
```

Use `with_block_config(shape, kernel_config, blocks)` to tune the kernel too.
Blocks are the atomic unit: `shape()` describes the block grid and `block_count()` gives
its total number of blocks. `input_shape()` and `input_numel()` describe the tensor
accepted by `update`. Cell queries, `total_weight(idx)`, and merge selections all use flat
**block indices**, never original element indices. `total_weight` counts a block's pooled
observations.

When blocks pool multiple elements, updates process the input directly without retaining
full tensor sample buffers. Elementwise layouts—including a requested count at least as
large as the axis—retain normal buffering. Scratch memory scales with block size per active
worker. Block settings survive snapshot round-trips. Merging combines whole blocks using
their observation counts, including unequal-sized blocks. The visualizer displays the block
grid directly. Elementwise tracking is simply the special case of one-element blocks.
`block_size()` reports the requested size (`None` in balanced mode); `blocks_per_axis()`
reports the requested count in balanced mode and the effective count in size mode.

The snapshot format now records the grouping mode. Older RankKnot v5 / TDigest v4
snapshots must be regenerated; incompatible versions are rejected explicitly.

### Why RankKnot is the default

RankKnot is a compact streaming rank summary designed for tensors with many independently
tracked positions. For each position it retains at most 32 weighted `f32` knots, 16-bit
probability masses, a mask for retained exact repeated-value intervals, and exact minimum
and maximum sidecars. The knot summary occupies **208 bytes per position**, plus an 8-byte observation counter, compared with
approximately **4,900 bytes** for the default TDigest configuration. Updates are buffered in
256-row batches and positions are compressed independently in parallel with Rayon.

In the initial ten-workload accuracy suite, RankKnot had lower mean and maximum rank error
than TDigest on nine workloads; TDigest won the 95%-zero activation case. A local Apple M4
run measured about 78% less retained heap and 81% less peak heap for RankKnot. Tensor-wide
RankKnot merges had lower mean and maximum error than TDigest on all ten representative
workloads.

These are initial results, not universal guarantees: adversarial workloads have mixed
winners, and throughput depends on tensor shape and platform. See the
[RankKnot algorithm specification](https://github.com/vladim0105/monatq/blob/master/docs/rankknot.md)
for the algorithm, invariants, complexity, complete initial results, reproduction commands,
and limitations.

### Kernels

| Kernel | Element types | Contract |
| --- | --- | --- |
| `RankKnot` *(default)* | `f32`, `i32` | complete; 208 B of state per position |
| `TDigest` | `f32`, `i32` | complete; ~4,900 B of state per position |

Every kernel is selected statically, so there is no runtime dispatch cost. Name one
explicitly to override the default:

```rust
use monatq::{TDigest, TensorDigest};

let mut digest = TensorDigest::<f32, TDigest>::new(&[3, 4]);
```

Both kernels summarise `i32` at `f32` resolution, so integer magnitudes above 2^24
round to the nearest representable value.

### Errors

Fallible calls return `monatq::Result<T>`. Queries that cannot fail — `quantile`,
`quantiles`, `flush`, `block_count`, `shape` — stay infallible, so they need no `?`.

```rust
use monatq::{Error, TensorDigest};

let mut digest = TensorDigest::<f32>::new(&[3, 4]);

match digest.update(&[1.0, 2.0]) {
    Ok(()) => { /* ... */ }
    // Samples must match the tensor's element count.
    Err(error @ Error::ShapeMismatch { .. }) => eprintln!("{error}"),
    Err(error) => return Err(error),
}
```

`Error` also distinguishes `InvalidSnapshot` (the bytes were read fine but do not
describe a usable digest) from `Io` (the file or device failed), and carries
`ShapeMismatch`, `IndexOutOfBounds`, and `InvalidConfig`.

NaN input is a documented precondition rather than a checked one: `update` does not
validate it, and a later flush will panic.

### Snapshots

```rust
use monatq::TensorDigest;

let mut digest = TensorDigest::<f32>::new(&[3, 4]);
// ... update the digest ...

// Serialize to memory and restore with a known kernel and element type.
let bytes = digest.to_bytes()?;
let restored = TensorDigest::<f32>::from_bytes(&bytes)?;

// Or detect both the kernel and the element type from the snapshot itself.
let restored_any = monatq::from_bytes(&bytes)?;
println!("{} over {}", restored_any.kernel_name(), restored_any.dtype_name());
```

`to_bytes` uses the same zstd-compressed bincode snapshot format as `save`, so file and
in-memory snapshots are interchangeable. Snapshots are self-describing: `monatq::from_bytes`
and `monatq::load` return an `AnyTensorDigest` identifying the kernel and element type, while
the typed loaders still reject a snapshot written by a different kernel rather than
reinterpreting its state. Each kernel uses one current versioned format for both element-wise and
blockwise tracking. Older snapshot formats are rejected; regenerate existing snapshots.
Snapshots contain summary state, not ingestion workspace: TDigest reconstructs its row buffer
and derived capacities on load, while RankKnot stores its per-block states directly.

## Features

- **Parallel updates** - element-wise compression runs in parallel via Rayon
- **Custom T-Digest** - optimised implementation for the tensor case, making per-position quantile tracking practical at tensor scale
- **Distribution analysis** - classify each position as Normal, Uniform, Laplace, or LogNormal by fitting an empirical quantile profile
- **Snapshots** - zstd-compressed bincode snapshots via file-based `save` / `load` or in-memory `to_bytes` / `from_bytes`, versioned and validated on load
- **Typed errors** - fallible calls return `monatq::Result`; infallible ones stay infallible
- **Visualisation** - built-in HTTP server (`digest.visualize()`) for browser-based inspection of a tensor

## License

Apache-2.0 - see [LICENSE](LICENSE).
