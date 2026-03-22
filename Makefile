.PHONY: install wheel

## Build and install the Python bindings into the active Python environment.
## Run this after any changes to the Rust source to pick up the latest version.
install:
	cd monatq-py && maturin develop --release

## Build a release wheel (.whl) for the current platform into monatq-py/dist/.
wheel:
	cd monatq-py && maturin build --release --out dist
