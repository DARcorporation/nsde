#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('nsde/__init__.py').read(),
)[0]

setup(
    name="nsde",
    version=__version__,
    description="Non-dominated Sorting Differential Evolution Algorithm",
    author="Daniel de Vries",
    author_email="danieldevries6@gmail.com",
    packages=["nsde"],
    install_requires=[
        "numpy>=1.17",
        "openmdao<2.10,>=2.8",
        "tqdm<5,>=4.32",
        "six<1.13,>=1.12",
        "nsga2-utils==0.0.5",
    ],
    tests_require=[
        "parameterized",
    ],
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
    ],
    license="MIT License",
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],
)
