# monatq

**Monakhov Tensor Quantiles** - approximate quantile tracking for tensors using T-Digest.

`monatq` maintains one T-Digest per element position across many tensor samples, enabling fast approximate quantile queries (median, percentiles, etc.) over the observed value distribution at each position. Updates are parallelised element-wise via Rayon.

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

// Track a [3, 4] tensor (12 elements)
let mut digest = TensorDigest::new(&[3, 4], 100);

// Feed samples (row-major flat slices)
for sample in my_tensor_samples {
    digest.update(&sample);
}

// Query the per-element median
let medians: Vec<f32> = digest.quantile(0.5);

// Query multiple quantiles at once
let [p10, p50, p90] = digest.quantiles(&[0.1, 0.5, 0.9])[..] else { panic!() };

// Classify the distribution shape at each position
let distributions = digest.analyze();
```

### Snapshots

```rust
use monatq::TensorDigest;

let mut digest = TensorDigest::<f32>::new(&[3, 4], 100);
// ... update the digest ...

// Serialize to memory and restore with a known element type.
let bytes = digest.to_bytes().unwrap();
let restored = TensorDigest::<f32>::from_bytes(&bytes).unwrap();

// Or detect f32/i32 from the embedded dtype tag.
let restored_any = monatq::from_bytes(&bytes).unwrap();
```

`to_bytes` uses the same zstd-compressed bincode snapshot format as `save`, so file and
in-memory snapshots are interchangeable.

## Features

- **Parallel updates** - element-wise compression runs in parallel via Rayon
- **Custom T-Digest** - optimised implementation for the tensor case, making per-position quantile tracking practical at tensor scale
- **Distribution analysis** - classify each position as Normal, Uniform, Laplace, or LogNormal by fitting an empirical quantile profile
- **Snapshots** - zstd-compressed bincode snapshots via file-based `save` / `load` or in-memory `to_bytes` / `from_bytes`
- **Visualisation** - built-in HTTP server (`digest.visualize()`) for browser-based inspection of a tensor

## License

Apache-2.0 - see [LICENSE](LICENSE).
