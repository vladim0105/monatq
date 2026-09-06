# monatq (Python)

Python bindings for [monatq](https://github.com/vladim0105/monatq) — approximate quantile tracking for tensors with element-wise or axis-local block updates.

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

Use `BlockConfig` to choose either a fixed size or a requested count of groups along one
axis. The constructor is keyword-only and requires exactly one mode:

```python
from monatq import BlockConfig, TensorDigest

# Fixed-size groups: 64, 64, then a short final group of 1.
digest = TensorDigest(
    shape=[256, 129, 2],
    blocks=BlockConfig(block_size=64, axis=1),
)
assert digest.input_shape == [256, 129, 2]  # original input shape
assert digest.shape == [256, 3, 2]          # atomic block grid
assert digest.block_size == 64
assert digest.blocks_per_axis == 3          # effective group count

# A requested count instead makes balanced groups (9 values, then groups of 8).
balanced = TensorDigest(
    shape=[256, 129, 2],
    blocks=BlockConfig(blocks_per_axis=16, axis=1),
)
assert balanced.shape == [256, 16, 2]
assert balanced.block_size is None
assert balanced.blocks_per_axis == 16       # requested count
```

Both kernels support both modes. `axis` defaults to `-1` and accepts negative indices,
just like Rust's `BlockConfig`. The shared Rust layout resolves and validates the axis;
`digest.block_axis` reports the resolved nonnegative index, including after snapshot loading.
Blocks are one-dimensional and local to each axis-line: they never cross another axis.
Every raw value contributes directly to its block digest; values are not averaged first.
For fixed-size mode, the last group is shorter when the axis length has a remainder. For
count mode, group lengths differ by at most one. Requested counts above the axis length
produce elementwise groups, while `blocks_per_axis=0` explicitly selects elementwise
tracking.

The legacy `blocks_per_axis=` and `block_axis=` arguments remain supported. A convenience
`block_size=` argument can likewise be paired with `block_axis=`:

```python
TensorDigest([256, 129, 2], block_size=64, block_axis=1)
```

Do not combine `blocks=...` with any legacy/convenience block arguments, or specify both
size and count modes; conflicting inputs raise `ValueError`.

`update` accepts the complete original tensor described by `input_shape` and `input_numel`.
Every downstream operation uses blocks: quantile and analysis outputs contain one entry
per block, cell queries accept flat block indices, and merge selections identify whole
blocks. Merging uses actual observation counts, including unequal-sized blocks, and the
visualizer displays the block grid.

When blocks pool multiple elements, updates do not retain full tensor sample buffers.
Elementwise layouts retain normal buffering. Snapshots preserve the block mode and its
settings for both kernels.

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
