#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from os import path
import sys
import setuptools

here = path.abspath(path.dirname(__file__))

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open(path.join(here, "nsde/__init__.py")).read(),
)[0]


with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "nsde.sorting",
        ["src/sorting_module.cpp", "src/sorting.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            "src/include",
        ],
        language="c++",
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    # pybind11 2.4.x seems to be incompatible with c++17
    # flags = ["-std=c++17", "-std=c++14", "-std=c++11"]
    flags = ["-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        opts.append("-O2")
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name="nsde",
    version=__version__,
    description="Non-dominated Sorting Differential Evolution (NSDE) Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel de Vries",
    author_email="danieldevries6@gmail.com",
    packages=["nsde"],
    ext_modules=ext_modules,
    python_requires=">=3.6, <4",
    install_requires=[
        "numpy>=1.17",
        "tqdm<5,>=4.32",
        "six<1.13,>=1.12",
        "pybind11>=2.4",
    ],
    setup_requires=["pybind11>=2.4"],
    extras_require={"openmdao": "openmdao<2.10,>=2.8"},
    tests_require=["pytest", "parameterized"],
    cmdclass={"build_ext": BuildExt},
    url="https://github.com/daniel-de-vries/nsde",
    download_url="https://github.com/daniel-de-vries/nsde/archive/v{0}.tar.gz".format(
        __version__
    ),
    keywords=[
        "optimization",
        "black-box",
        "data science",
        "differential",
        "evolution",
        "evolutionary",
        "algorithms",
        "non-dominated",
        "crowding distance",
        "single-objective",
        "multi-objective",
    ],
    license="MIT License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
