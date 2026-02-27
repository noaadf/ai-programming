from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mytensor",  # Python 中 import mytensor
        sources=[
            "bindings.cpp",
            "tensor.cpp",
            "activation.cpp",
            "layers.cpp",
            "gemm.cpp",
        ],
        language="c++",
    ),
]

setup(
    name="mytensor",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
