# Experiments using Rust for parts of captain2

Trying to implement small parts of [captain2](https://github.com/captain-project/captain2) in Rust to look for potential performance improvements.

## Installation

*Note: I haven't double-checked these instructions yet, and haven't tried them on other OSes than Debian Linux.*

- [Install Rust](https://rustup.rs/)
- Install [maturin](https://github.com/PyO3/maturin): see [Installation - Maturin User Guide](https://www.maturin.rs/installation); I have a fork with somewhat verified dependencies that I use [here](https://github.com/pflanze/maturin), clone it, then inside it run: `cargo install --locked --path .`
- Install patchelf (" Try `pip install maturin[patchelf]` (or just `pip install patchelf`) "?). On Debian: `apt-get install patchelf`.
- numpy, python3-sparse can be installed from packages on Debian: `apt-get install python3-numpy-dev python3-sparse`

## Build

### Run the Rust-only tests

Run:

```shell
cargo test --release
```

If there are any errors, run (but note that this disables slow tests,
hopefully the errors weren't in those; otherwise re-add the
`--release` option and if necessary change the Cargo.toml file to add
debug information to release builds):

```shell
RUST_BACKTRACE=1 cargo test
```

### Build the library to use from Python

```shell
python3 -m venv --system-site-packages .venv
source `pwd`/.venv/bin/activate
maturin develop --release
./test
```

See [Distribution - Maturin User Guide](https://www.maturin.rs/distribution) for alternaties to the local installation approach above (virtual env and `maturin develop --release`).

