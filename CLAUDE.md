# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build          # Build the project
cargo test           # Run all tests
cargo test <name>    # Run a single test by name
cargo clippy         # Lint
cargo fmt            # Format code
```

## Project Overview

`monatq` (Monakhov Tensor Quantiles) is a Rust library for analysing tensors by tracking the distribution of values at each element position. The core idea is to maintain a [Kernel](https://github.com/tdunning/t-digest) per tensor element, allowing approximate quantile queries over the observed distribution at each position across many tensor samples.

For visual analysis of results, a web-based interface is used. It must be lightweight and portable (no heavy frameworks or server dependencies).

Performance is a primary concern:
- Element-wise updates are parallelised across tensor positions (e.g. via Rayon).
- The TDigest implementation is custom (not an off-the-shelf crate) to meet performance requirements. Avoid replacing it with a third-party TDigest library.

## Error handling

Fallible public calls return `monatq::Result<T>`; see `src/error.rs`. Two rules keep this
consistent:

- Operations with no failure mode stay infallible (`quantile`, `quantiles`, `flush`,
  `numel`, `shape`). Do not wrap them in `Result` for uniformity's sake.
- A kernel that does not implement an operation returns `Error::Unsupported` rather than
  `unimplemented!()`. Tooling that iterates over backends relies on this to skip a kernel
  instead of crashing the run. TDigest and RankKnot implement the whole contract; only
  Quantile Spine still reports `Unsupported`.

Shared, kernel-independent logic belongs outside the kernels. Distribution classification
lives in `src/distribution.rs` and takes a quantile closure; the HTTP visualizer in
`src/server/` is generic over `StorageOperations`. Neither should gain a kernel-specific
branch — if a kernel can answer rank queries and merge, it gets both for free.

NaN input is a documented precondition of `update`, not a checked one; it is not validated
on the hot path and panics during a later flush.

## Project Structure

This is a Rust workspace using the 2024 edition: the `monatq` library (entry point
`monatq/src/lib.rs`) and `monatq-py` PyO3 bindings. Kernels live in `monatq/src/kernels/`
and are selected statically through the sealed `DigestKernel` trait; not every kernel
implements every operation.

`TensorDigest<T, K>` defaults to `K = RankKnot`, and the Python constructor defaults to
`kernel="rankknot"`. TDigest remains fully supported and is selected explicitly. Snapshots
are self-describing: `monatq::from_bytes` detects kernel and dtype, while typed loaders
reject a snapshot from another kernel.

The Python bindings reify the (element type, kernel) matrix into a four-arm `Inner` enum
dispatched by the `dispatch!` / `dispatch_rewrap!` macros. Add a method to `Inner` using
those macros rather than writing a fresh four-way `match`.

Note that `cargo build -p monatq-py` fails at link time outside maturin (undefined
`_PyExc_*`); use `cargo check -p monatq-py` to validate the bindings. To run the Python
tests: `cd monatq-py && maturin develop --release && python -m pytest tests/`. The `torch`
tests are skipped unless torch is installed.
