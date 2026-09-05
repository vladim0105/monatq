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

### Why RankKnot is the default

RankKnot is a compact streaming rank summary designed for tensors with many independently
tracked positions. For each position it retains at most 32 weighted `f32` knots, 16-bit
probability masses, a mask for retained exact repeated-value intervals, and exact minimum
and maximum sidecars. This summary state occupies **208 bytes per position**, compared with
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
`quantiles`, `flush`, `numel`, `shape` — stay infallible, so they need no `?`.

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
reinterpreting its state.

## Features

- **Parallel updates** - element-wise compression runs in parallel via Rayon
- **Custom T-Digest** - optimised implementation for the tensor case, making per-position quantile tracking practical at tensor scale
- **Distribution analysis** - classify each position as Normal, Uniform, Laplace, or LogNormal by fitting an empirical quantile profile
- **Snapshots** - zstd-compressed bincode snapshots via file-based `save` / `load` or in-memory `to_bytes` / `from_bytes`, versioned and validated on load
- **Typed errors** - fallible calls return `monatq::Result`; infallible ones stay infallible
- **Visualisation** - built-in HTTP server (`digest.visualize()`) for browser-based inspection of a tensor

## License

Apache-2.0 - see [LICENSE](LICENSE).
