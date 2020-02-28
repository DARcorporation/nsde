# -*- coding: utf-8 -*-
"""Definition of several crossover strategies."""
import numpy as np


class CrossoverStrategy:
    """
    Abstract crossover strategy.
    """

    def __call__(self, parent, mutant, crossover_probability, rng):
        """Generate a new chromosome based on a parent and a mutant donor.

        Parameters
        ----------
        parent : array_like
            Chromosome of the parent
        mutant : array_like
            Chromosome of the mutant
        crossover_probability : float
            Characteristic probability of the crossover operation
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        np.array
            Offspring chromosome
        """
        raise NotImplementedError


class bin(CrossoverStrategy):
    """
    Binomial crossover strategy.
    """

    def __call__(self, parent, mutant, crossover_probability, rng):
        n = len(parent)
        p = rng.uniform(size=n) < crossover_probability
        if not np.any(p):
            p[rng.integers(n)] = True
        return np.where(p, mutant, parent)


class exp(CrossoverStrategy):
    """
    Exponential crossover strategy.
    """

    def __call__(self, parent, mutant, crossover_probability, rng):
        child = np.copy(parent)
        n = len(parent)
        idx = rng.integers(n, endpoint=True)
        for i in range(n):
            child[i] = mutant[i]
            idx = (idx + 1) % n
            if rng.uniform() >= crossover_probability:
                break
        return child


__strategies__ = {"bin": bin, "exp": exp}
