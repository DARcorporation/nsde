# -*- coding: utf-8 -*-
"""Definition of several mutation strategies."""
import numpy as np


class MutationStrategy:
    """
    Abstract mutation strategy.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def _mutation_helper(self, parent_idx, population, rng):
        """Helper function for the mutation strategies

        Parameters
        ----------
        n : int
            Number of mutation strategy (as in: 'rand/1', 'rand/2', etc.)
        parent_idx : int
            Index of the parent
        population : array_like
            Individuals of the population
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        idxs : np.array
            List of indices of chosen individuals to use for the mutation operation
        r : np.array
            List of chosen individuals to use for the mutation operation
        s : np.array
            Chromosome contribution shared by all mutation strategies
        """
        idxs = [idx for idx in range(population.shape[0]) if idx != parent_idx]
        idxs = rng.choice(idxs, size=1 + 2 * self.n, replace=False)
        r = population[idxs]
        s = np.sum(r[1:-1:2] - r[2: len(r) + 1: 2], axis=0)
        return idxs, r, s

    @staticmethod
    def _repair_f_cr(f, cr):
        """
        Ensure mutated mutation and crossover rates are always within a given range.

        Parameters
        ----------
        f : array_like
            Mutation rate(s)
        cr : array_like
            Crossover rate(s)

        Returns
        -------
        f : array_like
            Repaired mutation rate(s)
        cr : array_like
            Repaired crossover rate(s)
        """
        f = np.clip(f, 0.1, 1.0)
        cr = np.clip(cr, 0.0, 1.0)
        return f, cr

    def __call__(self, parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        """
        Create a mutated chromosome based on a parent donor and any properties related to fitness and dominance.

        Parameters
        ----------
        parent_idx : int
            Index of the parent
        population : array_like
            List of chromosomes of all individuals (2D array, with one row per individual)
        fitness : array_like
            List of the fitness of each individual (2D array, with one column per objective)
        fronts : array_like
            List of sets of indices of non-dominated fronts
        f : array_like
            Mutation rates
        cr : array_like
            Crossover rates
        rng : numpy.random.Generator
            Random number generator
        self_adaptive : 0, 1, or 2
            0 for no, 1 for simple, and 2 for complex self-adaptivity

        Returns
        -------
        mutant : np.ndarray
            Chromosome of a mutant
        f : float
            Mutant mutation rate
        cr : float
            Mutant crossover rate
        """
        raise NotImplementedError


class rand(MutationStrategy):
    """
    The 'rand/n' mutation strategy
    """

    def __call__(self, parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        idxs, r, s = self._mutation_helper(parent_idx, population, rng)

        if self_adaptive:
            f_mutant = f[idxs[0]] + np.sum(
                [
                    rng.normal() * 0.5 * (f[idxs[i]] - f[idxs[i + 1]])
                    for i in range(1, self.n + 1)
                ]
            )
            cr_mutant = cr[idxs[0]] + np.sum(
                [
                    rng.normal() * 0.5 * (cr[idxs[i]] - cr[idxs[i + 1]])
                    for i in range(1, self.n + 1)
                ]
            )

            f_mutant, cr_mutant = self._repair_f_cr(f_mutant, cr_mutant)
        else:
            f_mutant = f[parent_idx]
            cr_mutant = cr[parent_idx]

        mutant = r[0] + f_mutant * s
        return mutant, f_mutant, cr_mutant


class best(MutationStrategy):
    """
    The 'best/n' mutation strategy
    """

    def __call__(self, parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        idxs, r, s = self._mutation_helper(parent_idx, population, rng)

        if fronts is None:
            i_best = np.argmin(fitness)
        else:
            i_best = rng.choice(fronts[0])

        if self_adaptive:
            f_mutant = f[i_best] + np.sum(
                [
                    rng.normal() * 0.5 * (f[idxs[i]] - f[idxs[i + 1]])
                    for i in range(1, self.n + 1)
                ]
            )
            cr_mutant = cr[i_best] + np.sum(
                [
                    rng.normal() * 0.5 * (cr[idxs[i]] - cr[idxs[i + 1]])
                    for i in range(1, self.n + 1)
                ]
            )

            f_mutant, cr_mutant = self._repair_f_cr(f_mutant, cr_mutant)
        else:
            f_mutant = f[parent_idx]
            cr_mutant = cr[parent_idx]

        mutant = population[i_best] + f_mutant * s
        return mutant, f_mutant, cr_mutant


class rand_to_best(MutationStrategy):
    """
    The 'rand-to-best/n' mutation strategy
    """

    def __call__(self, parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        idxs, r, s = self._mutation_helper(parent_idx, population, rng)

        if fronts is None:
            i_best = np.argmin(fitness)
        else:
            i_best = rng.choice(fronts[0])

        if self_adaptive:
            f_mutant = (
                f[parent_idx]
                + rng.normal() * 0.5 * (f[i_best] - f[parent_idx])
                + np.sum(
                    [
                        rng.normal() * 0.5 * (f[idxs[i]] - f[idxs[i + 1]])
                        for i in range(1, self.n + 1)
                    ]
                )
            )
            cr_mutant = (
                cr[parent_idx]
                + rng.normal() * 0.5 * (cr[i_best] - cr[parent_idx])
                + np.sum(
                    [
                        rng.normal() * 0.5 * (cr[idxs[i]] - cr[idxs[i + 1]])
                        for i in range(1, self.n + 1)
                    ]
                )
            )

            f_mutant, cr_mutant = self._repair_f_cr(f_mutant, cr_mutant)
        else:
            f_mutant = f[parent_idx]
            cr_mutant = cr[parent_idx]

        mutant = r[0] + f_mutant * ((population[i_best] - r[0]) + s)
        return mutant, f_mutant, cr_mutant


__strategies__ = {"rand": rand, "best": best, "rand-to-best": rand_to_best}
