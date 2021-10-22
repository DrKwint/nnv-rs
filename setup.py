from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="nnv-rs",
    version="0.2.0",
    rust_extensions=[
        RustExtension("nnv_rs.nnv_rs", binding=Binding.PyO3, native=True)
    ],
    packages=["nnv_rs"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)