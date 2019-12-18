#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of the EvolutionStrategy class and related helper functions."""
import numpy as np


def _mutation_helper(n, parent_idx, population, rng):
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
    idxs = rng.choice(idxs, size=1 + 2 * n, replace=False)
    r = population[idxs]
    s = np.sum(r[1:-1:2] - r[2 : len(r) + 1 : 2], axis=0)
    return idxs, r, s


def rand(n):
    """The 'rand/n' mutation strategy

    Parameters
    ----------
    n : int
        Number of the mutation strategy (as in: 'rand/1', 'rand/2', etc.)

    Returns
    -------
    callable
        Mutation function
    """

    def mutate(parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        idxs, r, s = _mutation_helper(n, parent_idx, population, rng)

        if self_adaptive:
            f_mutant = f[idxs[0]] + np.sum(
                [
                    rng.normal() * 0.5 * (f[idxs[i]] - f[idxs[i + 1]])
                    for i in range(1, n)
                ]
            )
            cr_mutant = cr[idxs[0]] + np.sum(
                [
                    rng.normal() * 0.5 * (cr[idxs[i]] - cr[idxs[i + 1]])
                    for i in range(1, n)
                ]
            )
        else:
            f_mutant = f[parent_idx]
            cr_mutant = cr[parent_idx]

        mutant = r[0] + f_mutant * s
        return mutant, f_mutant, cr_mutant

    return mutate


def best(n):
    """The 'best/n' mutation strategy

    Parameters
    ----------
    n : int
        Number of the mutation strategy (as in: 'best/1', 'best/2', etc.)

    Returns
    -------
    callable
        Mutation function
    """

    def mutate(parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        idxs, r, s = _mutation_helper(n, parent_idx, population, rng)

        if fronts is None:
            i_best = np.argmin(fitness)
        else:
            i_best = rng.choice(fronts[0])

        if self_adaptive:
            f_mutant = f[i_best] + np.sum(
                [
                    rng.normal() * 0.5 * (f[idxs[i]] - f[idxs[i + 1]])
                    for i in range(1, n)
                ]
            )
            cr_mutant = cr[i_best] + np.sum(
                [
                    rng.normal() * 0.5 * (cr[idxs[i]] - cr[idxs[i + 1]])
                    for i in range(1, n)
                ]
            )
        else:
            f_mutant = f[parent_idx]
            cr_mutant = cr[parent_idx]

        mutant = population[i_best] + f_mutant * s
        return mutant, f_mutant, cr_mutant

    return mutate


def rand_to_best(n):
    """'The 'rand-to-best/n' mutation strategy

    Parameters
    ----------
    n : int
        Number of the mutation strategy (as in: 'rand-to-best/1', 'rand-to-best/2', etc.)

    Returns
    -------
    callable
        Mutation function
    """

    def mutate(parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive):
        idxs, r, s = _mutation_helper(n, parent_idx, population, rng)

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
                        for i in range(1, n)
                    ]
                )
            )
            cr_mutant = (
                cr[parent_idx]
                + rng.normal() * 0.5 * (cr[i_best] - cr[parent_idx])
                + np.sum(
                    [
                        rng.normal() * 0.5 * (cr[idxs[i]] - cr[idxs[i + 1]])
                        for i in range(1, n)
                    ]
                )
            )
        else:
            f_mutant = f[parent_idx]
            cr_mutant = cr[parent_idx]

        mutant = r[0] + f_mutant * ((population[i_best] - r[0]) + s)
        return mutant, f_mutant, cr_mutant

    return mutate


def bin(parent, mutant, crossover_probability, rng):
    """The binomial crossover strategy

    Parameters
    ----------
    parent : array_like
        Chromosome of the parent
    mutant : array_like
        Chromosome of the mutant
    crossover_probability : float
        Probability, applied to each gene individually, that the mutant's gene will replace the parent's
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.array
        Offspring chromosome
    """
    n = len(parent)
    p = rng.uniform(size=n) < crossover_probability
    if not np.any(p):
        p[rng.integers(n)] = True
    return np.where(p, mutant, parent)


def exp(parent, mutant, crossover_probability, rng):
    """The exponential crossover strategy

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
    child = np.copy(parent)
    n = len(parent)
    idx = rng.integers(n, endpoint=True)
    for i in range(n):
        child[i] = mutant[i]
        idx = (idx + 1) % n
        if rng.uniform() >= crossover_probability:
            break
    return child


def clip(mutant, rng):
    """The clipping repair strategy

    This strategy clips each out of bounds gene to its nearest bound.

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
    return np.clip(mutant, 0, 1)


def random(mutant, rng):
    """The random repair strategy

    This strategy selects a uniform random value for each out of bounds gene.

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
    loc = np.logical_or(mutant < 0, mutant > 1)
    count = np.sum(loc)
    if count > 0:
        np.place(mutant, loc, rng.uniform(size=count))
    return mutant


class EvolutionStrategy:
    """An evolution strategy such as 'rand/1/exp'"""

    __mutation_strategies__ = {"rand": rand, "best": best, "rand-to-best": rand_to_best}
    __crossover_strategies__ = {"bin": bin, "exp": exp}
    __repair_strategies__ = {"clip": clip, "random": random}

    def __init__(self, designation):
        """Create a evolution strategy from a designation.

        A traditional designation, such as 'rand/1/bin' is accepted.
        However, an additional part for the chromosome repair strategy
        can be added to the end as well (should be one of 'clip' or 'random').
        If this part is omitted, the 'clip' strategy will be used by default.

        Parameters
        ----------
        designation : str
            Evolution strategy designation, such as 'rand/1/bin/clip'.

        Raises
        ------
        ValueError
            If the designation is invalid.
        """
        try:
            self._designation = designation.lower()
            parts = self._designation.split("/")
            self.mutate = self.__mutation_strategies__[parts[0]](int(parts[1]))
            self.crossover = self.__crossover_strategies__[parts[2]]
            if len(parts) >= 4:
                self.repair = self.__repair_strategies__[parts[3]]
            else:
                self.repair = clip
                self._designation += "/clip"
        except AttributeError or KeyError or IndexError or ValueError:
            raise ValueError(f"Invalid evolution strategy '{designation}'")

    def __call__(
        self, parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive=False
    ):
        """Procreate!

        Parameters
        ----------
        parent_idx : int
            Index of the target parent in the population
        population : array_like
            List of individuals making up the population
        fitness : array_like
            Fitness of the individuals in the population
        fronts : list of lists of ints or None
            List the indices of individuals for each pareto front or None if single-objective
        f : float or array_like
            Mutation rate
        cr : float or array_like
            Crossover probability
        rng : np.random.Generator
            Random number generator
        self_adaptive : bool
            True for self-adaptivity

        Returns
        -------
        np.array
            A child!
        """
        mutant, f_mutant, cr_mutant = self.mutate(
            parent_idx, population, fitness, fronts, f, cr, rng, self_adaptive
        )
        mutant = self.crossover(population[parent_idx], mutant, cr_mutant, rng)
        mutant = self.repair(mutant, rng)
        return mutant, f_mutant, cr_mutant

    def __repr__(self):
        return self._designation
