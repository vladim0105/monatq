# monatq (Python)

Python bindings for [monatq](https://github.com/vladim0105/monatq) — approximate quantile tracking for tensors using T-Digest, with element-wise parallel updates.

## Installation

```bash
pip install monatq
```

To build and install from source into the active Python environment (requires [maturin](https://github.com/PyO3/maturin)):

```bash
make install
```

## Usage

The bindings accept NumPy arrays and PyTorch tensors directly (float32, CPU, contiguous).

```python
from monatq import TensorDigest

digest = TensorDigest(shape=[3, 4])    # kernel="rankknot" by default

for tensor in my_tensors:              # torch.Tensor or np.ndarray, shape [3, 4]
    digest.update(tensor)

medians  = digest.quantile(0.5)        # list of 12 floats
p10, p90 = digest.quantiles([0.1, 0.9])
labels   = digest.analyze()            # e.g. ["Normal", "Uniform", ...]

digest.save("checkpoint.mq")
digest = TensorDigest.load("checkpoint.mq")   # kernel and dtype are detected
```

### Kernels

`kernel` selects the summary structure. `"rankknot"` is the default and uses far less memory
per tensor position; `"tdigest"` is the original.

```python
TensorDigest(shape=[3, 4])                                  # rankknot
TensorDigest(shape=[3, 4], buffer_capacity=512)             # rankknot, tuned
TensorDigest(shape=[3, 4], kernel="tdigest", compression=100)
```

The tuning knobs are kernel-specific: `compression` belongs to `"tdigest"` and
`buffer_capacity` to `"rankknot"`. Passing one to the wrong kernel raises `ValueError`
rather than being silently ignored.

Everything after `shape` is keyword-only. Code written against the previous
`TensorDigest(shape, compression)` positional form needs
`TensorDigest(shape, kernel="tdigest", compression=...)`.

### Snapshots

```python
blob    = digest.to_bytes()
restored = TensorDigest.from_bytes(blob)
restored.kernel        # "rankknot"
restored.dtype         # "float32"
```

### Sparse tensors

`without_zeros()` returns a copy with values at zero removed, which is useful when exact
zeros dominate a distribution and hide its shape. Quantiles of the result describe the
nonzero subpopulation; `count` is carried over unchanged and no longer matches it.

## Use Cases

- **Model interpretability** - feed activation or weight tensors through a forward pass and query per-position quantiles to understand how individual neurons or channels behave across inputs.
- **Quantization & pruning** - characterise the value distribution at each position to inform bit-width selection, clipping ranges, or sparsity thresholds without storing all observations in memory.

![monatq visualizer](https://raw.githubusercontent.com/vladim0105/monatq/master/example.png)

## License

Apache-2.0 - see [LICENSE](https://github.com/vladim0105/monatq/blob/master/LICENSE).
