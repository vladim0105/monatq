# monatq

**Monakhov Tensor Quantiles** - approximate quantile tracking for tensors.

`monatq` provides a unified `TensorDigest<T, K>` container with statically selected `TDigest` and `QuantileSpine` kernels. Each kernel retains its optimized flat storage layout, and updates are parallelised element-wise via Rayon.

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
use monatq::{TDigest, TensorDigest};

// Track a [3, 4] tensor (12 elements) with the T-Digest kernel and default config.
let mut digest = TensorDigest::<f32, TDigest>::new(&[3, 4]);

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

Select Quantile Spine without runtime dispatch:

```rust
use monatq::{QuantileSpine, TensorDigest};

let mut digest = TensorDigest::<f32, QuantileSpine>::new(&[3, 4]);
```

### Errors

Fallible calls return `monatq::Result<T>`. Queries that cannot fail — `quantile`,
`quantiles`, `flush`, `numel`, `shape` — stay infallible, so they need no `?`.

```rust
use monatq::{Error, QuantileSpine, TensorDigest};

let mut digest = TensorDigest::<f32, QuantileSpine>::new(&[3, 4]);

match digest.merge_all() {
    Ok(merged) => { /* ... */ }
    // Not every kernel implements every operation. This is a property of the
    // kernel, not of the data: it will never start succeeding.
    Err(error @ Error::Unsupported { .. }) => eprintln!("{error}"),
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
use monatq::{TDigest, TensorDigest};

let mut digest = TensorDigest::<f32, TDigest>::new(&[3, 4]);
// ... update the digest ...

// Serialize to memory and restore with a known element type.
let bytes = digest.to_bytes()?;
let restored = TensorDigest::<f32, TDigest>::from_bytes(&bytes)?;

// Or detect f32/i32 from the embedded dtype tag.
let restored_any = monatq::from_bytes(&bytes)?;
```

`to_bytes` uses the same zstd-compressed bincode snapshot format as `save`, so file and
in-memory snapshots are interchangeable. Each snapshot records which kernel wrote it, so
handing a RankKnot snapshot to the t-digest loader is a clean `InvalidSnapshot` error
rather than a misparse.

## Features

- **Parallel updates** - element-wise compression runs in parallel via Rayon
- **Custom T-Digest** - optimised implementation for the tensor case, making per-position quantile tracking practical at tensor scale
- **Distribution analysis** - classify each position as Normal, Uniform, Laplace, or LogNormal by fitting an empirical quantile profile
- **Snapshots** - zstd-compressed bincode snapshots via file-based `save` / `load` or in-memory `to_bytes` / `from_bytes`, versioned and validated on load
- **Typed errors** - fallible calls return `monatq::Result`; infallible ones stay infallible
- **Visualisation** - built-in HTTP server (`digest.visualize()`) for browser-based inspection of a tensor

## License

Apache-2.0 - see [LICENSE](LICENSE).
