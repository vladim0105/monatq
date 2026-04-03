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

digest = TensorDigest(shape=[3, 4], compression=100)

for tensor in my_tensors:              # torch.Tensor or np.ndarray, shape [3, 4]
    digest.update(tensor)

medians  = digest.quantile(0.5)        # list of 12 floats
p10, p90 = digest.quantiles([0.1, 0.9])
labels   = digest.analyze()           # e.g. ["Normal", "Uniform", ...]

digest.save("checkpoint.mq")
digest = TensorDigest.load("checkpoint.mq")
```

## Use Cases

- **Model interpretability** - feed activation or weight tensors through a forward pass and query per-position quantiles to understand how individual neurons or channels behave across inputs.
- **Quantization & pruning** - characterise the value distribution at each position to inform bit-width selection, clipping ranges, or sparsity thresholds without storing all observations in memory.

![monatq visualizer](https://raw.githubusercontent.com/vladim0105/monatq/master/example.png)

## License

Apache-2.0 - see [LICENSE](https://github.com/vladim0105/monatq/blob/master/LICENSE).
