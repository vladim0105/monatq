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

### Blockwise tracking

For quantization, choose how many balanced groups to track along one axis:

```python
digest = TensorDigest(shape=[256, 129, 2], blocks_per_axis=16, block_axis=1)
assert digest.input_shape == [256, 129, 2] # original input shape
assert digest.shape == [256, 16, 2]       # atomic block grid
assert digest.block_count == 8192
```

Along each axis-line, the first block pools 9 raw values and the remaining 15 pool 8 each.
Both kernels support this. `blocks_per_axis=0` is the default and means elementwise;
1 pools the whole selected axis. Counts above the axis length clamp to that length.
`block_axis` defaults to the last axis and accepts negative indices. Blocks never cross
other axes. Every raw value contributes—values are not averaged before tracking.

The block count applies independently to each combination of the other coordinates, not
to the tensor as a whole. For the example above, choosing `block_axis=-1` would clamp to
2 blocks on the last axis and therefore produce elementwise tracking.

`update` accepts the complete original tensor described by `input_shape` and `input_numel`.
Every downstream operation uses blocks: quantile and analysis outputs contain one entry
per block, cell queries accept flat block indices, and merge selections identify whole
blocks. Merging uses actual observation counts, including unequal-sized blocks, and the
visualizer displays the block grid.

When blocks pool multiple elements, updates do not retain full tensor sample buffers.
Elementwise layouts retain normal buffering, including when the requested count is at
least the axis length. Snapshots preserve block settings.

### Snapshots

```python
blob    = digest.to_bytes()
restored = TensorDigest.from_bytes(blob)
restored.kernel        # "rankknot"
restored.dtype         # "float32"
```

Only the current snapshot format is supported. Regenerate snapshots written by older builds.

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
