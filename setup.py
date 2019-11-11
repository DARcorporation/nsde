#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('differential_evolution/__init__.py').read(),
)[0]

setup(
    name="differential_evolution",
    version=__version__,
    description="Differential Evolution Algorithm with OpenMDAO Driver",
    author="Daniel de Vries",
    author_email="danieldevries6@gmail.com",
    packages=["differential_evolution"],
    install_requires=[
        "numpy>=1.17",
        "openmdao<2.10,>=2.8",
        "tqdm<5,>=4.32",
        "six<1.13,>=1.12",
    ],
    tests_require=[
        "parameterized",
    ],
    url="https://github.com/daniel-de-vries/differential-evolution",
    download_url="https://github.com/daniel-de-vries/differential-evolution/archive/v{0}.tar.gz".format(
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
