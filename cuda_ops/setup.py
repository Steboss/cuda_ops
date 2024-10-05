#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sysconfig
import numpy
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CUDAExtension(Extension):
    """This is a setuptools extension that handles building CUDA files."""

    def __init__(self, name, sources, *args, **kwargs):
        """Constructor for the CUDAExtension class."""
        super().__init__(name, sources, *args, **kwargs)
        self.sources = sources


class custom_build_ext(build_ext):
    """Custom build extension to handle CUDA and nvcc"""

    def initialize_options(self):
        super().initialize_options()
        self.inplace = True

    def build_extensions(self):
        """Build the CUDA extensions"""

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                # compile the .cu CUDA file
                self.compiler.set_executable("compiler_so", "nvcc")
                include_dirs = self.compiler.include_dirs
                # this is aligned with the Makefile
                nvcc_args = [
                    "-c",
                    src,
                    "-o",
                    obj,
                    "-Xcompiler",
                    "-fPIC",
                ]
                nvcc_args += ["-I" + inc for inc in include_dirs]
                nvcc_args += extra_postargs
                self.compiler.spawn(["nvcc"] + nvcc_args)
            else:
                # default compiler
                super_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler.src_extensions.append(".cu")
        default_compiler_so = self.compiler.compiler_so
        super_compile = self.compiler._compile

        try:
            self.compiler._compile = custom_compile
            # python and numpy include
            self.compiler.include_dirs.extend(
                [numpy.get_include(), sysconfig.get_paths()["include"]]
            )
            build_ext.build_extensions(self)
        finally:
            self.compiler._compile = super_compile
            self.compiler.compiler_so = default_compiler_so


ext_modules = [
    CUDAExtension(
        name="cuda_ops.rms_norm",
        sources=[
            "src/rms_norm.cu",
        ],
        include_dirs=["/usr/local/cuda/include"],
        library_dirs=["/usr/local/cuda/lib64"],
        extra_compile_args={},
        language="c++",
    )
]

setup(
    name="cuda_ops",
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext},
    install_requires=["numpy"],
    extras_require={
        "test": [
            "pytest",
        ]
    },
    include_package_data=True,
    package_data={"cuda_ops": ["*.so"], "test": ["*"]},
)
