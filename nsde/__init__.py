#!/usr/bin/env python
# -*- coding: utf-8 -*-
__version__ = "0.0.12"

from .nsde import NSDE
from .strategies import EvolutionStrategy

try:
    from .openmdao import NSDEDriver
except ModuleNotFoundError:
    pass
