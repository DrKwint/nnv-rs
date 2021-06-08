# nnv-rs

### Build Python package

1. Install Rust with `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and re-launch your shell.
2. Create a Python virtualenv and run `pip install -r requirements.txt` to install Python requirements.
3. Clone the following Rust projects `ndarray-linalg`, `numpy`, `truncnorm-rs`
4. Set the `ndarray` version in `numpy` to `15.2.0`
5. Ensure `CMake` 3.15 or higher is installed as well as `clang`
6. Install OpenBlas (`sudo apt-get install libopenblas-dev` on Ubuntu)
6. Switch to nightly Rust with `rustup default nightly`
7. Build and install the Rust-backed Python package with `python ./setup.py install`

### Troubleshooting

- Build gives linker error `/usr/bin/ld: cannot find -lCbcSolver`: cbc solver is a default dependency of the `good_lp` package we're using for linear programming. Fix on Ubuntu is to run `sudo apt install coinor-libcbc-dev`.