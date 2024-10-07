#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sysconfig

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import ScmVersion


class CUDAExtension(Extension):
    """This is a setuptools extension that handles building CUDA files."""

    def __init__(self, name, sources, *args, **kwargs):
        """Constructor for the CUDAExtension class."""
        super().__init__(name, sources, *args, **kwargs)
        self.sources = sources


class custom_build_ext(build_ext):  # noqa N801
    """Custom build extension to handle CUDA and nvcc"""

    def initialize_options(self):
        """Constructors for the custom build options"""
        super().initialize_options()
        self.inplace = True

    def build_extensions(self):
        """Build the CUDA extensions"""

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            """This is the core function for compiling CUDA files.

            Args:
                obj (str): The object file to be created
                src (str): The source file to be compiled
                ext (Extension): The extension object
                cc_args (List[str]): The compiler arguments
                extra_postargs (List[str]): Extra arguments to be passed to the compiler
                pp_opts (List[str]): Preprocessor options
            """
            if src.endswith(".cu"):
                self.compiler.set_executable("compiler_so", "nvcc")
                include_dirs = self.compiler.include_dirs
                # align the command to the Makefile
                nvcc_args = ["-I" + inc for inc in include_dirs]
                nvcc_args += [
                    "-Xcompiler",
                    "-fPIC",
                    "-c",
                    "-o",
                    obj,
                    src,
                ]
                nvcc_args += extra_postargs
                self.compiler.spawn(["nvcc"] + nvcc_args)
            else:
                super_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler.src_extensions.append(".cu")
        default_compiler_so = self.compiler.compiler_so
        super_compile = self.compiler._compile

        try:
            self.compiler._compile = custom_compile
            # python and numpy include
            self.compiler.include_dirs.extend([numpy.get_include(), sysconfig.get_paths()["include"]])
            build_ext.build_extensions(self)
        finally:
            self.compiler._compile = super_compile
            self.compiler.compiler_so = default_compiler_so


def resolve_version():
    """This function retrieves the correct version for the package

    The function checks the env variables to retrieve info about the
    state.
    If we are on PR we will use the default versioning scheme.
    If we are merging on main, then we will have a STABLE RELEASE variable
    set to 1. This avoid to add the local version to the version number.
    """

    def my_release_branch_semver_version(version: ScmVersion):
        if os.getenv("STABLE_RELEASE"):
            return os.getenv("BASE_VERSION")

        if os.getenv("DEV_VERSION"):
            return os.getenv("DEV_VERSION")

        return version.format_with("{tag}.dev0")

    version_dictionary = {"version_scheme": my_release_branch_semver_version}
    if os.getenv("STABLE_RELEASE") or os.getenv("DEV_VERSION"):
        version_dictionary["local_scheme"] = "no-local-version"

    return version_dictionary


ext_modules = [
    CUDAExtension(
        name="cuda_ops.rms_norm",
        sources=[
            "src/rms_norm.cu",
        ],
        include_dirs=["/usr/local/cuda/include"],
        library_dirs=["/usr/local/cuda/lib64"],
        libraries=["cudart"],
        extra_compile_args={},
        extra_link_args=[],
        language="c++",
    )
]

setup(
    use_scm_version=resolve_version(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext},
    include_package_data=True,
    package_data={"cuda_ops": ["*.so"], "test": ["*"]},
)
