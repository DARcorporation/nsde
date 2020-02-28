# -*- coding: utf-8 -*-
"""Definition of the EvolutionStrategy class."""
import numpy as np

from . import mutation, crossover, repair


class EvolutionStrategy:
    """An evolution strategy such as 'rand/1/exp'"""

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
            self.mutate: mutation.MutationStrategy = mutation.__strategies__[parts[0]](int(parts[1]))
            self.crossover: crossover.CrossoverStrategy = crossover.__strategies__[parts[2]]()
            if len(parts) >= 4:
                self.repair: repair.RepairStrategy = repair.__strategies__[parts[3]]()
            else:
                self.repair: repair.RepairStrategy = repair.clip()
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
