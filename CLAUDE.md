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

## Project Structure

This is a Rust library crate using the 2024 edition. The entry point is `src/lib.rs`. There are currently no external dependencies.
