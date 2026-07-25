# RankKnot feasibility experiment

This directory contains the reproducible Python feasibility test for the RankKnot compact tensor-history backend proposed in `docs/rankknot-whitepaper.typ`.

## Run

```bash
python3 docs/experiments/transport_coreset_feasibility.py \
  > docs/experiments/results/transport-coreset-holdout.csv
```

Requirements: Python 3.11+, NumPy, and Cargo. The script creates the Rust baseline crate and binary inputs under the system temporary directory.

## Compared layouts

- **RTC-48-stream:** 48 `f32` locations with `u32` counts, streamed in 256-row batches.
- **RTC-64-u16-stream / RankKnot:** 64 `f32` locations and 64 prefix-rounded `u16` probability masses in a 400-byte summary-state target layout. This figure excludes the batch input buffer and flush workspace.
- **RTC-48-offline:** one-shot compression of the full empirical distribution; this is a representation upper bound, not a streaming algorithm.
- Repository **TDigest** and **QuantileSpine**, run through `TensorDigest` on identical ordered `f32` values.

The RankKnot paper additionally proposes exact `f32` minimum and maximum sidecars. They replace the per-cell count because the tensor sample count is shared by the parent `TensorDigest`. The current rank grid excludes probabilities zero and one, so these sidecars do not alter the reported interior results.

## Protocol

- Held-out seeds: 44 and 55; exploratory choices used seed 40.
- 32,768 values per workload.
- 256-row streaming flushes.
- 235 tie-aware rank queries from 0.0001 through 0.9999.
- Smooth, heavy-tailed, multimodal, atomic, quantized, constant, and order-diagnostic workloads.

## Important limitations

- This is an algorithmic Python/f64 prototype with persistent representatives rounded to `f32`; it does not predict Rust throughput or complete live allocation.
- The 400-byte figure covers summary state only. At 256 rows, an `f32` implementation also needs 1,024 input-buffer bytes per tensor position. A Rust implementation should use worker-local sorting scratch rather than another tensor-sized 1,024-byte-per-position sorted copy.
- The `u16` state stores normalized masses. Prefix rounding controls each current CDF boundary, but long-stream requantization drift has not been tested.
- The offline row cannot be used as a streaming performance claim.
- No real downstream consumer has yet been validated. PTQ is one example; grouped summaries, visualizations, and threshold decisions also need testing.
