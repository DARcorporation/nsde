#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

exec(open("differential_evolution/version.py").read())

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
        "pytest",
        "pytest-pep8",
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
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.7",
    ],
)
