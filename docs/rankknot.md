# RankKnot

RankKnot is the default quantile kernel in `monatq` 0.3. It is a fixed-memory streaming summary for tracking an approximate empirical distribution independently at every tensor position.

This document describes the current K32 Rust implementation. It is an implementation specification with initial measurements, not a claim of a universal error bound or a formal publication.

## Status

RankKnot implements the complete `TensorDigest` contract:

- `f32` and `i32` ingestion;
- per-position quantile queries and exact summary extrema;
- cell, channel, and tensor-wide merging;
- distribution analysis and zero filtering;
- versioned snapshots; and
- the optional HTTP visualizer.

The public default is:

```rust
use monatq::TensorDigest;

let mut digest = TensorDigest::<f32>::new(&[3, 4]);
```

Use an explicit configuration to change the number of buffered rows:

```rust
use monatq::{RankKnotConfig, TensorDigest};

let mut digest = TensorDigest::<f32>::with_config(
    &[3, 4],
    RankKnotConfig { buffer_capacity: 512 },
);
```

The knot count, rank scale, and mass encoding are internal constants rather than public configuration.

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

It is not intended to provide a distribution-free worst-case error guarantee. Accuracy depends on the stream, query probabilities, buffer capacity, and repeated recompression.

## State representation

Each position stores:

| Field | Type | Bytes | Meaning |
| --- | --- | ---: | --- |
| `values` | `[f32; 32]` | 128 | Active representatives in ascending order |
| `masses` | `[u16; 32]` | 64 | Probability masses totaling 65,535 |
| `pure_mask` | `u64` | 8 | Marks representatives that retain one exact value interval |
| `min`, `max` | two `f32` values | 8 | Exact encoded endpoints |
| **Total** |  | **208** | No per-position pointers or heap allocations |

One `u64` sample count is shared by the containing storage because every position receives one observation per successful tensor update.

The default input buffer holds 256 complete tensor rows. For `f32`, that adds 1,024 bytes per position while collecting. Worker-local sorting and merge vectors are temporary and scale with active Rayon work rather than tensor width.

Both supported input types are summarized at `f32` resolution. An `i32` magnitude above 2^24 may therefore round to the nearest representable `f32`; TDigest has the same crate-level output limitation.

## Update algorithm

`update` first checks that the sample contains exactly `numel` values. A shape mismatch returns `Error::ShapeMismatch` without modifying the digest. Valid samples are copied into the row buffer.

Setting `buffer_capacity` to `0` bypasses the input buffer and updates immediately on every sample.

When the buffer reaches a positive `buffer_capacity`, or a query explicitly flushes it, each tensor position is processed independently in parallel:

1. Gather the position's buffered column into worker-local `f32` scratch.
2. Sort the incoming values.
3. Linearly merge them with the position's existing weighted support.
4. Coalesce equal values and preserve purity only when every contribution is pure.
5. Place up to 31 boundaries at fixed tail-companded target ranks.
6. Store an exact value for a pure singleton group or an `f64`-accumulated weighted mean for a mixed group.
7. Prefix-round cumulative probability onto the 0–65,535 mass scale using ties-to-even.
8. Update the exact encoded extrema.

With the default configuration, compression sees at most 256 new values plus 32 existing representatives per position.

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

### Mass encoding

Cumulative mass is rounded before adjacent prefixes are differenced:

```text
prefixⱼ = round_ties_even(65,535 × cumulativeⱼ / total)
massⱼ   = prefixⱼ - prefixⱼ₋₁
```

This directly controls every represented CDF boundary and ensures active encoded masses sum to 65,535. Repeated compression can still accumulate approximation error.

## Query algorithm

A query flushes pending rows first.

For a nonempty state:

- `q <= 0` returns `min`;
- `q >= 1` returns `max`;
- a NaN probability returns NaN;
- a target inside a pure knot's mass interval returns that exact value; and
- mixed representatives are anchored at their mass-center ranks and interpolated linearly.

The resulting quantile curve is monotone because values and rank anchors are ordered. Interpolation adjacent to an infinity returns the infinity instead of producing NaN.

An empty state returns `0.0` for every probability. This check occurs before the NaN-probability check.

## Merging

`merge_cells`, `merge_channels`, and `merge_all` treat each selected summary as positive weighted support:

1. Emit every active knot and encoded mass.
2. Sort the union once.
3. Coalesce equal values.
4. Run the same compression routine used during ingestion.
5. Union the exact extrema.

All positions in one storage share the same sample count, so their encoded masses are already on a common scale. The merged sample count is the source sample count multiplied by the number of selected positions. A merged digest can continue accepting updates.

Merging is lossy because it recompresses approximate support. The initial measurements below are encouraging, but repeated merge-of-merge chains have not been characterized.

## Zero filtering

`without_zeros` removes knots located at zero and recompresses the survivors. Its quantiles describe the nonzero subpopulation.

The storage-wide sample count is carried over unchanged because the number of nonzero observations can differ by position. A filtered digest should therefore be treated as a distribution shape to inspect, not as a reliable nonzero population count.

## Snapshots

Snapshots contain a RankKnot kernel tag, format version, knot count, mass scale, dtype, shape, sample count, and the flat state arrays. They are bincode-encoded and zstd-compressed.

Loading validates:

- kernel, format, dtype, knot count, and mass scale;
- array lengths against the shape;
- ascending active values;
- absence of NaN knots; and
- active masses summing to either zero or 65,535.

`buffer_capacity` is deliberately not persisted because it changes ingestion behavior rather than the encoded distribution. A loaded digest starts with the default capacity.

The current snapshot format revision is 1. Encoding constants are implementation details; a future build that changes them must reject incompatible snapshots rather than reinterpret their masses.

## Invariants

The compressor maintains:

- active representatives ordered by `f32::total_cmp`;
- zero mass for unused slots;
- total active mass of 65,535 for a nonempty state;
- no NaN active representatives;
- purity bits only for retained exact-value intervals;
- exact encoded extrema for supported input; and
- a quantile curve that cannot decrease as probability increases.

The snapshot loader explicitly checks ordering, active mass normalization, and the absence of NaN knots before exposing decoded state. Endpoint queries return the stored extrema.

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
| Snapshot encode/decode | `O(PK)` | Flat serialized arrays |

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
| Normal | 0.000458 / 0.001460 | 0.000627 / 0.006950 |
| Uniform | 0.000681 / 0.001170 | 0.001334 / 0.007380 |
| Log-normal | 0.000860 / 0.002290 | 0.001831 / 0.012570 |
| Exponential | 0.000840 / 0.001940 | 0.001623 / 0.010610 |
| Laplace | 0.000700 / 0.001880 | 0.000775 / 0.007120 |
| Overlapping bimodal | 0.000469 / 0.001450 | 0.000896 / 0.007580 |
| 32-level normal | 0.000094 / 0.000660 | 0.009511 / 0.034780 |
| 50% zeros | 0.000380 / 0.001380 | 0.026949 / 0.252720 |
| 95% zero activations | 0.000508 / 0.006160 | 0.000359 / 0.004750 |
| Heterogeneous tensor | 0.000511 / 0.002260 | 0.003170 / 0.051750 |

RankKnot had lower mean and maximum error than TDigest on nine of the ten representative workloads. The 95%-zero workload is the counterexample.

The adversarial report uses 65,536 samples at one position and 1,003 probabilities. Its winners are mixed; no universal ordering is claimed.

### Memory

Heap figures come from an instrumented global allocator and exclude input data, exact truth, and query outputs. The following 32-position measurements were recorded on the local Apple M4 run used for the 0.3 documentation:

| Backend | Retained after flush | Ingestion peak |
| --- | ---: | ---: |
| RankKnot | 39,432 B | 45,064 B |
| TDigest | 182,408 B | 234,328–239,208 B |

RankKnot used about 78% less retained heap and 81% less peak heap than TDigest in this run. These allocator totals include input buffers, shape vectors, headers, and worker scratch; they are intentionally larger than the 208-byte summary-state figure.

### Tensor-wide merging

When all 32 positions were merged and compared with the exact pooled population, RankKnot had lower mean and maximum rank error than TDigest on all ten representative workloads.

Selected results:

| Pooled workload | RankKnot mean / max | TDigest mean / max |
| --- | ---: | ---: |
| Normal | 0.000426 / 0.001220 | 0.000753 / 0.001970 |
| Log-normal | 0.000859 / 0.001984 | 0.004540 / 0.020050 |
| 50% zeros | 0.000362 / 0.001130 | 0.033803 / 0.250106 |
| Heterogeneous tensor | 0.000978 / 0.002736 | 0.007044 / 0.026593 |

The merged RankKnot digest retained 1,240 bytes versus TDigest's 5,708 bytes. Merge peak allocation was 17,880 bytes versus approximately 37–38 KB.

### Throughput

A local Apple M4 Divan run used each backend's default configuration:

| Update workload | RankKnot median | TDigest median | RankKnot relative result |
| --- | ---: | ---: | ---: |
| 64×64 × 1,000 normal samples | 8.73 ms | 7.72 ms | 13% slower |
| 64×64 × 1,000 uniform samples | 8.81 ms | 7.51 ms | 17% slower |
| 256×256 × 200 uniform samples | 21.47 ms | 26.81 ms | 20% faster |

These timings are platform-specific. They show that RankKnot's compact state does not imply a universal throughput win: it trailed TDigest on the smaller repeated-flush cases and led on the larger tested tensor.

## Limitations and open work

- There is no distribution-free accuracy bound for the fixed 32-knot state.
- Long-stream mass drift beyond the current tests remains to be measured.
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
