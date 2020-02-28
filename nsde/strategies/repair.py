# -*- coding: utf-8 -*-
"""Definition of several repair strategies."""
import numpy as np


class RepairStrategy:
    """
    Abstract repair strategy.
    """

    def __call__(self, mutant, rng):
        """
        Repair the chromosome of a mutant.

        Parameters
        ----------
        mutant : array_like
            Chromosome of the mutant
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        np.array
            Repaired chromosome
        """
        raise NotImplementedError


class clip(RepairStrategy):
    """
    Clipping repair strategy.

    This strategy clips each out of bounds gene to its nearest bound.
    """

    def __call__(self, mutant, rng):
        return np.clip(mutant, 0, 1)


class random(RepairStrategy):
    """
    Random repair strategy.

    This strategy selects a uniform random value for each out of bounds gene.
    """

    def __call__(self, mutant, rng):
        loc = np.logical_or(mutant < 0, mutant > 1)
        count = np.sum(loc)
        if count > 0:
            np.place(mutant, loc, rng.uniform(size=count))
        return mutant


__strategies__ = {"clip": clip, "random": random}
