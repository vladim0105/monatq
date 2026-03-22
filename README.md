# monatq

**Monakhov Tensor Quantiles** — approximate quantile tracking for tensors using T-Digest.

`monatq` maintains one T-Digest per element position across many tensor samples, enabling fast approximate quantile queries (median, percentiles, etc.) over the observed value distribution at each position. Updates are parallelised element-wise via Rayon.

## Installation

```toml
[dependencies]
monatq = "0.1"
```

## Usage

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

## Features

- **Parallel updates** — element-wise compression runs in parallel via Rayon
- **Custom T-Digest** — optimised SoA layout with a three-phase merge loop; no third-party digest dependency
- **Distribution analysis** — classify each position as Normal, Uniform, Laplace, LogNormal, or Unknown
- **Save / load** — zstd-compressed bincode snapshots via `digest.save(path)` / `TensorDigest::load(path)`
- **Visualisation** — built-in HTTP server (`digest.visualize()`) for browser-based inspection

## License

Apache-2.0 — see [LICENSE](LICENSE).
