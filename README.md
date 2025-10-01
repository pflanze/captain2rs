# Experiments using Rust for parts of captain2

Trying to implement small parts of [captain2](https://github.com/captain-project/captain2) in Rust to look for potential performance improvements.

## Installation

- [Install Rust](https://rustup.rs/)
- Install [maturin](https://github.com/PyO3/maturin): see [Installation - Maturin User Guide](https://www.maturin.rs/installation); I have a fork with somewhat verified dependencies that I use [here](https://github.com/pflanze/maturin), clone it, then inside it run: `cargo install --locked --path .`
- Install patchelf (" Try `pip install maturin[patchelf]` (or just `pip install patchelf`) "?)
- numpy, python3-sparse can be installed from packages on Debian

## Build

### Rust-only tests

From the toplevel of the clone:

```shell
~/captain2rs$ cargo run --release --bin main
```

### To use from Python

From the nested captain2rs subdirectory:

```shell
~/captain2rs/captain2rs$ python3 -m venv --system-site-packages .venv
~/captain2rs/captain2rs$ source `pwd`/.venv/bin/activate
~/captain2rs/captain2rs$ maturin develop --release
~/captain2rs/captain2rs$ ./test
```

See [Distribution - Maturin User Guide](https://www.maturin.rs/distribution) for alternaties to the local installation approach above (virtual env and `maturin develop --release`).

