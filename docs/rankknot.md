# RankKnot

RankKnot is the default quantile kernel in `monatq` 0.3. It is a fixed-memory streaming summary for tracking an approximate empirical distribution independently at every tensor position.

This document describes the current K32 Rust implementation. It is an implementation specification with initial measurements, not a claim of a universal error bound or a formal publication.

## Status

RankKnot implements the complete `TensorDigest` contract with blocks as the atomic unit:

- `f32` and `i32` ingestion;
- per-position quantile queries and exact summary extrema;
- cell, channel, and tensor-wide merging;
- distribution analysis and zero filtering;
- the optional HTTP visualizer.

The public default is:

```rust
use monatq::TensorDigest;

let mut digest = TensorDigest::<f32>::new(&[3, 4]);
```

Use an explicit configuration to change how many new values each block collects before compression:

```rust
use monatq::{RankKnotConfig, TensorDigest};

let mut digest = TensorDigest::<f32>::with_config(
    &[3, 4],
    RankKnotConfig { buffer_capacity: 512 },
);
```

The knot count and rank scale are internal constants rather than public configuration.

## Problem

A `TensorDigest` receives complete row-major tensor samples. It tracks a separate distribution for every flat tensor position so that quantiles and grouping decisions can be made after collection.

Keeping every observation is usually impractical for large tensors. A conventional digest per position also carries substantial state. RankKnot instead keeps at most 32 weighted support locations per position while preserving exact encoded ties and extrema.

This supports decisions that can be expressed through marginal rank distributions. It does **not** retain:

- raw sample identity or temporal order;
- correlations between tensor positions;
- exact arbitrary interior quantiles; or
- enough information to reconstruct the original observations.

## Design goals

RankKnot is designed for:

- compact state that scales across many tensor positions;
- batched, parallel ingestion;
- useful resolution in both tails;
- deterministic behavior;
- explicit treatment of repeated values;
- exact minimum and maximum queries at the summary's `f32` resolution;
- merging without replaying the original samples; and
- a small query-time surface with no per-position allocation.

It is not intended to provide a distribution-free worst-case error guarantee. Accuracy depends on the stream, query probabilities, and repeated recompression. Because knot weights are exact counts, it does not depend on `buffer_capacity`.

## State representation

Each position stores:

| Field | Type | Bytes | Meaning |
| --- | --- | ---: | --- |
| `values` | `[f32; 32]` | 128 | Active representatives in ascending order |
| `counts` | `[u32; 32]` | 128 | Observation count represented by each knot |
| `pure_mask` | `u64` | 8 | Marks representatives that retain one exact value interval |
| `min`, `max` | two `f32` values | 8 | Exact encoded endpoints |
| **Total** |  | **272** | No per-position pointers or heap allocations |

Each statistical state also has an 8-byte `u64` observation counter. Knot counts sum to this counter unless a state was halved to fit `u32` (see below) or filtered by `without_zeros`. Knot counts sum to this counter unless a state was halved to fit `u32` (see below) or filtered by `without_zeros`. A separate storage-wide `u64` counts accepted tensor samples. These differ in blockwise mode: a block receives one observation per contained element per tensor sample. Balanced block sizes differ by at most one.

The input buffer holds enough complete tensor rows for each full-size block to collect `buffer_capacity` new values: `ceil(buffer_capacity / block_len)` rows. With one-element blocks and the default capacity, that is 16 rows, or 64 bytes per position for `f32`. When one row already fills a block, or `buffer_capacity` is zero, no buffer is allocated. Worker-local sorting and merge vectors are temporary and scale with the incoming values per block per active Rayon worker.

Total memory is approximately `block_count × (280 + 4 × buffer_capacity)` bytes, where the buffer term is zero when no buffer is allocated. It excludes the caller's input tensor and worker scratch.

Both supported input types are summarized at `f32` resolution. An `i32` magnitude above 2^24 may therefore round to the nearest representable `f32`; TDigest has the same crate-level output limitation.

## Update algorithm

`update` first checks that the sample contains exactly `numel` values. A shape mismatch returns `Error::ShapeMismatch` without modifying the digest. Valid samples are copied into the row buffer.

When no buffer is allocated, each sample is compressed immediately. Otherwise, when the buffer is full, or a query explicitly flushes it, each block is processed independently in parallel:

1. Gather the block's buffered values into worker-local `f32` scratch.
2. Sort the incoming values.
3. Linearly merge them with the position's existing weighted support.
4. Coalesce equal values and preserve purity only when every contribution is pure.
5. Place up to 31 boundaries at fixed tail-companded target ranks.
6. Store an exact value for a pure singleton group or an `f64`-accumulated weighted mean for a mixed group.
7. Store each group's exact summed count as its knot count.
8. Update the exact encoded extrema.

With the default configuration, each compression of a full-size block sees at least 16 new values plus 32 existing representatives, whatever the block size.

### Blockwise ingestion

`TensorDigest::with_blocks(shape, BlockConfig::block_size(size, axis))` partitions the selected axis into fixed-width 1D groups, independently for each combination of the other coordinates. Size must be positive. The final group may be shorter; no padding is ingested. For axis length 129 and size 8, there are sixteen groups of 8 and a final group of 1. These are axis-local segments, not 2D tiles.

Alternatively, `BlockConfig::blocks_per_axis(count, axis)` requests balanced groups. The count must be positive and is clamped to the axis length; `BlockConfig::Elementwise` (the default) selects elementwise tracking. For axis length L and effective count B, the first L % B blocks have L / B + 1 elements and the remaining blocks have L / B elements. For L = 129 and B = 16, sizes are 9 followed by fifteen 8s.
Axes are signed in both Rust and Python: `-1` selects the last input dimension. The shared Rust layout resolves and validates the axis once; queries and snapshots use the normalized nonnegative index. Groups never cross the other axes. Buffering follows the same rule for every block length, as described above. All raw values enter the shared tracker, not their average. Each block's observation counter supplies the old population weight during compression.

`shape()` and `numel()` describe the original input geometry used by ingestion, exactly as passed at construction. `block_shape()` describes the atomic block grid and `block_count()` gives its total number of blocks. Bulk queries return one entry per block; cell queries and merge selections use flat block indices directly. Visualization displays the same block grid.

### Tail-companded boundaries

For slot coordinate `s` in `[0, 1]`, the target rank is:

```text
q(s) = sin²(πs / 2)
```

Uniform slot spacing in `s` places more boundaries near probability zero and one. A desired boundary is snapped to the nearest weighted entry boundary. Duplicate boundaries are skipped, so a large repeated value consumes one support location rather than many.

### Representatives

A group containing one pure input location retains that exact location and marks its purity bit. A mixed group stores its weighted mean:

```text
value(group) = Σ(weightᵢ × valueᵢ) / Σ(weightᵢ)
```

A mixed representative remains mixed even if rounding makes it equal to an observed value. This avoids inventing an exact tie.

### Exact counts

Each knot stores the exact number of observations its group represents. When old knots merge with new values, an old knot weighs its count and each new value weighs 1, so combining support is exact integer addition with no rounding. Small batches are therefore never rounded away, and accuracy does not depend on `buffer_capacity` or stream length. Approximation comes only from grouping entries into at most 32 representatives.

If a group's count would exceed `u32::MAX`, every count in the state is halved, rounding up so no knot reaches zero, until the largest fits. The observation counter keeps the true population, and on the next merge old knots are scaled back up by `observation count / count sum` so they are not underweighted against new values.

## Query algorithm

A query flushes pending rows first.

For a nonempty state:

- `q <= 0` returns `min`;
- `q >= 1` returns `max`;
- a NaN probability returns NaN;
- a target inside a pure knot's rank interval returns that exact value; and
- mixed representatives are anchored at their center ranks and interpolated linearly.

The resulting quantile curve is monotone because values and rank anchors are ordered. Interpolation adjacent to an infinity returns the infinity instead of producing NaN.

An empty state returns `0.0` for every probability. This check occurs before the NaN-probability check.

## Merging

`merge_cells`, `merge_channels`, and `merge_all` treat each selected summary as positive weighted support:

1. Emit every active knot with its count.
2. Sort the union once.
3. Coalesce equal values.
4. Run the same compression routine used during ingestion.
5. Union the exact extrema.

Selected blocks are weighted by their individual observation counts, so blocks with different sizes contribute their proper population. A merged digest can continue accepting updates. Individual input elements cannot be separated back out of a pooled block.

Merging is lossy because it recompresses approximate support. The initial measurements below are encouraging, but repeated merge-of-merge chains have not been characterized.

## Zero filtering

`without_zeros` removes knots located at zero and recompresses the survivors with their counts unchanged. Its quantiles describe the nonzero subpopulation.

The storage-wide sample count is carried over unchanged because the number of nonzero observations can differ by position. A filtered digest should therefore be treated as a distribution shape to inspect, not as a reliable nonzero population count.

## Snapshots

Snapshots contain a RankKnot kernel tag, format version, knot count, dtype, tensor sample count, block layout (the sole source of input and block geometry), per-block observation counts, and a vector of per-block states. Each state directly encodes its 32 values, 32 counts, purity mask, and extrema; no flattening into parallel vectors is needed. Snapshots are bincode-encoded and zstd-compressed.

Loading validates:

- kernel, format version, dtype, and knot count;
- block layout consistency and array lengths against the block count;
- ascending active values;
- absence of NaN knots; and
- knot counts summing to no more than the block's observation count.

`buffer_capacity` is deliberately not persisted because it changes ingestion behavior rather than the encoded distribution. A loaded digest starts with the default capacity.

The current snapshot format revision is 7, used for both element-wise and blockwise tracking. Earlier revisions are rejected; regenerate old snapshots. The knot count is validated so an incompatible state width cannot silently change the meaning of a summary.

## Invariants

The compressor maintains:

- active representatives ordered by `f32::total_cmp`;
- zero count for unused slots;
- knot counts summing to at most the observation count;
- no NaN active representatives;
- purity bits only for retained exact-value intervals;
- exact encoded extrema for supported input; and
- a quantile curve that cannot decrease as probability increases.

The snapshot loader explicitly checks ordering, count consistency, and the absence of NaN knots before exposing decoded state. Endpoint queries return the stored extrema.

Positive and negative infinity are supported as protected pure singleton groups. NaN observations are unsupported. They are intentionally not scanned during `update` and currently panic later when the buffered position is sorted.

## Complexity

Let:

- `P` be the number of tensor positions;
- `B` be `buffer_capacity`;
- `K = 32`; and
- `M` be the number of summaries selected for a merge.

Approximate costs are:

| Operation | Work | Additional storage |
| --- | --- | --- |
| `update` before flush | `O(P)` copy | Existing row buffer |
| Flush | `O(P × (B log B + B + K))`, parallel over positions | Worker-local `O(B + K)` scratch |
| One tensor-wide quantile | `O(P × K)`, parallel over positions | Output vector |
| Cell quantiles | `O(number of probabilities × K)` | Output vector |
| Merge `M` positions | `O(MK log(MK))` | `O(MK)` temporary support |
| Snapshot encode/decode | `O(PK)` | Encoded payload and decoded states; no intermediate flattened arrays |

`K` is fixed in the current implementation, but it is shown explicitly to describe the algorithm rather than only its present constant factors.

## Initial results

These measurements are initial implementation evidence, not universal accuracy or performance guarantees. Results depend on workload, configuration, platform, tensor width, Rayon scheduling, and system load.

### Accuracy protocol

`backend_accuracy` uses:

- 100,000 samples at each of 32 tensor positions;
- nine probabilities: 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, and 0.999;
- deterministic generated workloads; and
- tie-aware empirical rank-interval error.

An estimate receives zero error when the requested probability lies inside the empirical CDF jump for that returned value. Lower values are better.

| Workload | RankKnot mean / max | TDigest mean / max |
| --- | ---: | ---: |
| Normal | 0.000432 / 0.001280 | 0.000627 / 0.006950 |
| Uniform | 0.000815 / 0.001600 | 0.001334 / 0.007380 |
| Log-normal | 0.000929 / 0.002420 | 0.001831 / 0.012570 |
| Exponential | 0.000961 / 0.002010 | 0.001623 / 0.010610 |
| Laplace | 0.000744 / 0.001670 | 0.000775 / 0.007120 |
| Overlapping bimodal | 0.000426 / 0.001270 | 0.000896 / 0.007580 |
| 32-level normal | 0.000000 / 0.000000 | 0.009511 / 0.034780 |
| 50% zeros | 0.000572 / 0.014520 | 0.026949 / 0.252720 |
| 95% zero activations | 0.000466 / 0.006270 | 0.000359 / 0.004750 |
| Heterogeneous tensor | 0.000563 / 0.002240 | 0.003170 / 0.051750 |

RankKnot had lower mean and maximum error than TDigest on nine of the ten representative workloads. The 95%-zero workload is the counterexample.

The adversarial report uses 65,536 samples at one position and 1,003 probabilities. Its winners are mixed; no universal ordering is claimed.

### Memory

Heap figures come from an instrumented global allocator and exclude input data, exact truth, and query outputs. The following 32-position measurements were recorded on a local Apple M4 run with the default configuration:

| Backend | Retained after flush | Ingestion peak |
| --- | ---: | ---: |
| RankKnot | 11,024 B | 11,856 B |
| TDigest | 182,416 B | 233,536–239,216 B |

RankKnot used about 94% less retained heap and 95% less peak heap than TDigest in this run. These allocator totals include input buffers, shape vectors, headers, and worker scratch; they are intentionally larger than the 272-byte summary-state figure.

### Tensor-wide merging

When all 32 positions were merged and compared with the exact pooled population, RankKnot had lower mean and maximum rank error than TDigest on all ten representative workloads.

Selected results:

| Pooled workload | RankKnot mean / max | TDigest mean / max |
| --- | ---: | ---: |
| Normal | 0.000400 / 0.000965 | 0.000753 / 0.001970 |
| Log-normal | 0.000925 / 0.001856 | 0.004540 / 0.020050 |
| 50% zeros | 0.000314 / 0.000877 | 0.033803 / 0.250106 |
| Heterogeneous tensor | 0.001003 / 0.002752 | 0.007044 / 0.026593 |

The merged RankKnot digest retained 360 bytes versus TDigest's 5,716 bytes. Merge peak allocation was 17,000 bytes versus approximately 37–38 KB.

### Throughput

A local Apple M4 Divan run used each backend's default configuration:

| Update workload | RankKnot median | TDigest median | RankKnot relative result |
| --- | ---: | ---: | ---: |
| 64×64 × 1,000 normal samples | 31.37 ms | 7.58 ms | 4.1× slower |
| 64×64 × 1,000 uniform samples | 31.67 ms | 7.68 ms | 4.1× slower |
| 256×256 × 200 uniform samples | 72.42 ms | 27.77 ms | 2.6× slower |

These timings are platform-specific. RankKnot's small default buffer trades update throughput for memory: every 16 samples each position is recompressed. A larger `buffer_capacity` amortizes compression over more values and speeds up ingestion without changing accuracy.

## Limitations and open work

- There is no distribution-free accuracy bound for the fixed 32-knot state.
- `backend_accuracy` stops at 100,000 samples per position; longer streams are not part of the checked-in report.
- Repeated merge-of-merge chains are not characterized.
- Abrupt distribution shifts repeatedly approximate old state.
- Separated modes can expose interpolation across unsupported value gaps.
- Sparse activation behavior depends strongly on the exact zero fraction.
- NaN ingestion is an unchecked precondition and currently panics during flush.
- Cross-platform throughput artifacts are not checked in.
- No downstream quantizer or other consumer has been validated against RankKnot state.
- Knots remain crate-private; callers cannot export weighted support directly.

## Reproducing the results

Run the current accuracy, merge, adversarial, and allocator report:

```bash
cargo run -p monatq --release --bin backend_accuracy
```

Run update and query throughput benchmarks:

```bash
cargo bench -p monatq --bench tensor_digest
```

Run RankKnot tests:

```bash
cargo test -p monatq rankknot
cargo test -p monatq --test rankknot
cargo test -p monatq --test rankknot_analysis
```

For a release-quality comparison, record the commit, Rust version, target triple, CPU, thread count, and full command output alongside the results.
