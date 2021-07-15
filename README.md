nnv-rs
======

Fast reachability analysis and sampling for deep neural networks

Based loosely on the [matlab toolbox of a similar name](https://github.com/verivital/nnv) largely developed by [Dr. Hoang-Dung Tran](https://scholar.google.com/citations?user=_RzS3uMAAAAJ). I'm developing this software for my dissertation and I'm happy to support other work using it. If you're using this package or working on similar problems, please let me know by email at `equint at cse dot unl dot <educational ending>` (sorry, trying to thwart automated mailers).

Build Python package
--------------------

1. Install Rust with `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and re-launch your shell.
2. Create a Python virtualenv and run `pip install -r requirements.txt` to install Python requirements.
3. Clone the following Rust projects `ndarray-linalg`, `numpy`, `truncnorm-rs`
4. Set the `ndarray` version in `numpy` to `15.2.0`
5. Ensure `CMake` 3.15 or higher is installed as well as `clang`
6. Install OpenBlas (`apt-get install libopenblas-dev` on Ubuntu)
6. Switch to nightly Rust with `rustup default nightly`
7. Build and install the Rust-backed Python package with `python ./setup.py install`

Troubleshooting
---------------

- Build gives linker error `/usr/bin/ld: cannot find -lCbcSolver`: cbc solver is a default dependency of the `good_lp` package we're using for linear programming. Fix on Ubuntu is to run `sudo apt install coinor-libcbc-dev`.
- If your issue isn't listed here, open an issue on GitHub and we'll see if we can fix/add it.

Acknowledgements
----------------

This work is not supported by anybody.