.PHONY: install wheel test test-rust test-python check

## Check formatting, clippy, and compilation (including benches and tests)
check:
	cargo fmt --check
	cargo clippy --tests -- -D warnings
	cargo check --tests
	cargo check --benches

## Run Rust tests only
test-rust:
	cargo test

## Run Python binding tests only
test-python:
	cd monatq-py && python -m pytest tests/

## Run all tests (Rust + Python)
test: test-rust test-python

## Build and install the Python bindings into the active Python environment.
## Run this after any changes to the Rust source to pick up the latest version.
install:
	cd monatq-py && maturin develop --release

## Build a release wheel (.whl) for the current platform into monatq-py/dist/.
wheel:
	cd monatq-py && maturin build --release --out dist
