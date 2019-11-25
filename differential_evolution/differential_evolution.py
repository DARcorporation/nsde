#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of the Differential Evolution algorithm"""
import numpy as np

try:
    from openmdao.utils.concurrent import concurrent_eval
except ModuleNotFoundError:
    import warnings

    warnings.warn("OpenMDAO is not installed. Concurrent evaluation is not available.")

from .evolution_strategy import EvolutionStrategy


def mpi_fobj_wrapper(fobj):
    """
    Wrapper for the objective function to keep track of individual indices when running under MPI.

    Parameters
    ----------
    fobj : callable
        Original objective function

    Returns
    -------
    callable
        Wrapped function which, in addition to x, takes the individual's index and returns it along with f
    """

    def wrapped(x, ii):
        return fobj(x), ii

    return wrapped


class DifferentialEvolution:
    """
    Differential evolution algorithm.

    Attributes
    ----------
    fobj : callable
        Objective function
    lb, ub : array_like
        Lower and upper bounds
    range : array_like
        Distances between the lower and upper bounds
    f, cr : float
        Mutation rate and crossover probabilities
    adaptivity : int
        Method of self-adaptivity.
            - 0: No self-adaptivity. Specified mutation rate and crossover probability are used.
            - 1: Simple self-adaptability. Mutation rate and crossover probability are optimized Mote-Carlo style.
            - 2: Complex self-adaptability. Mutation rate and crossover probability are mutated with specified strategy.
    max_gen : int
        Maximum number of generations
    tolx, tolf : float
        Tolerances on the design vectors' and objective function values' spreads
    n_dim : int
        Number of dimension of the problem
    n_pop : int
        Population size
    rng : np.random.Generator
        Random number generator
    comm : MPI communicator or None
        The MPI communicator that will be used objective evaluation for each generation
    model_mpi : None or tuple
        If the model in fobj is also parallel, then this will contain a tuple with the the
        total number of population points to evaluate concurrently, and the color of the point
        to evaluate on this rank
    strategy : EvolutionStrategy
        Evolution strategy to use for procreation
    pop : np.array
        List of the individuals' chromosomes making up the current population
    fit : np.array
        Fitness of the individuals in the population
    best_idx, worst_idx : int
        Index of the best and worst individuals of the population
    best, worst : np.array
        Chromosomes of the best and worst individuals in the population
    best_fit, worst_fit : np.array
        Fitness of the best and worst individuals in the population
    generation : int
        Generation counter
    """

    def __init__(
        self,
        strategy=None,
        mut=0.85,
        crossp=1.0,
        adaptivity=0,
        max_gen=1000,
        tolx=1e-8,
        tolf=1e-8,
        n_pop=None,
        seed=None,
        comm=None,
        model_mpi=None,
    ):
        self.fobj = None

        self.lb, self.ub = None, None
        self.range = 0

        self.f = mut
        self.cr = crossp

        self.max_gen = max_gen
        self.tolx = tolx
        self.tolf = tolf

        self.n_dim = 0
        self.n_pop = n_pop

        self.rng = np.random.default_rng(seed)

        if adaptivity not in [0, 1, 2]:
            raise ValueError("self_adaptivity must be one of (0, 1, 2).")
        self.adaptivity = adaptivity

        self.comm = comm
        self.model_mpi = model_mpi

        self.strategy = strategy
        if strategy is None:
            self.strategy = EvolutionStrategy("rand-to-best/1/exp")

        self.pop = None
        self.fit = None
        self.best_idx, self.worst_idx = 0, 0
        self.best, self.worst = None, None
        self.best_fit, self.worst_fit = 0, 0
        self.dx, self.df = np.inf, np.inf

        self.generation = 0

        self._is_initialized = False

    def init(self, fobj, bounds, pop=None):
        """
        Initialize the algorithm.

        Parameters
        ----------
        fobj : callable
            Objective function
        bounds : list of 2-tuples
            List of (lower, upper) bounds
        pop : None or array_like, optional
            Initial population. If None, it will be created at random.
        """
        # Set default values for the mutation and crossover parameters
        if self.f is None or 0.0 > self.f > 1.0:
            self.f = 0.85
        if self.cr is None or 0.0 > self.cr > 1.0:
            self.cr = 1.0

        # Prepare the objective function and compute the bounds and variable range
        self.fobj = fobj if self.comm is None else mpi_fobj_wrapper(fobj)
        self.lb, self.ub = np.asarray(bounds).T
        self.range = self.ub - self.lb

        # Compute the number of dimensions
        self.n_dim = len(bounds)

        # Initialize a random population if one is not specified
        if pop is not None:
            self.n_pop = pop.shape[0]
            self.pop = pop
        else:
            if self.n_pop is None or self.n_pop <= 0:
                if self.comm is not None:
                    # If we are running under MPI, expand population to fully exploit all processors
                    self.n_pop = np.ceil(5 * self.n_dim / self.comm.size) * self.comm.size
                else:
                    # Otherwise, as a default, use 5 times the number of dimensions
                    self.n_pop = self.n_dim * 5

            self.pop = self.rng.uniform(self.lb, self.ub, size=(self.n_pop, self.n_dim))

        # Create random mutation/crossover parameters if self-adaptivity is used
        if self.adaptivity == 0:
            self.f = self.f * np.ones(self.n_pop)
            self.cr = self.cr * np.ones(self.n_pop)
        elif self.adaptivity == 1:
            self.f = self.rng.uniform(size=self.n_pop) * 0.9 + 0.1
            self.cr = self.rng.uniform(size=self.n_pop)
        elif self.adaptivity == 2:
            self.f = self.rng.uniform(size=self.n_pop) * 0.15 + 0.5
            self.cr = self.rng.uniform(size=self.n_pop) * 0.15 + 0.5

        # Ensure all processors have the same population and mutation/crossover parameters
        if self.comm is not None:
            self.pop, self.f, self.cr = self.comm.bcast((self.pop, self.f, self.cr), root=0)

        # Evaluate population fitness and update the class state
        self.fit = self(self.pop)
        self.update(self.pop, self.fit, self.f, self.cr)

        # Set generation counter to 0
        self.generation = 0

        # Mark class as initialized
        self._is_initialized = True

    @property
    def is_initialized(self):
        """bool: True if the algorithm has been initialized, False if not."""
        return self._is_initialized

    def __iter__(self):
        """
        This class is an iterator itself.

        Raises
        ------
        RuntimeError
            If this class is being used as an iterator before it has been initialized.
        """
        if not self._is_initialized:
            raise RuntimeError("DifferentialEvolution is not yet initialized.")
        return self

    def __next__(self):
        """
        Main iteration.

        Returns
        -------
        DifferentialEvolution
            The new state at the next generation.
        """
        if self.generation < self.max_gen and self.dx > self.tolx and self.df > self.tolf:
            # Create a new population and mutation/crossover parameters
            pop_new, f_new, cr_new = self.procreate()

            # Ensure all processors have the same updated population and mutation/crossover parameters
            if self.comm is not None:
                pop_new, f_new, cr_new = self.comm.bcast((pop_new, f_new, cr_new), root=0)

            # Evaluate the fitness of the new population
            fit_new = self(pop_new)

            # Update the class with the new data
            self.update(pop_new, fit_new, f_new, cr_new)

            # Compute spreads and update generation counter
            self.dx = np.linalg.norm(self.worst - self.best)
            self.df = np.abs(self.worst_fit - self.best_fit)
            self.generation += 1

            # Return the new state
            return self
        else:
            raise StopIteration

    def __call__(self, pop):
        """
        Evaluate the fitness of the given population.

        Parameters
        ----------
        pop : array_like
            List of chromosomes of the individuals in the population

        Returns
        -------
        np.array
            Fitness of the inviduals in the given population

        Notes
        -----
        If this class has an MPI communicator the individuals will be evaluated in parallel.
        Otherwise function evaluation will be serial.
        """
        # Evaluate generation
        if self.comm is not None:
            # Construct run cases
            cases = [((item, ii), None) for ii, item in enumerate(pop)]

            # Pad the cases with some dummy cases to make the cases divisible amongst the procs.
            extra = len(cases) % self.comm.size
            if extra > 0:
                for j in range(self.comm.size - extra):
                    cases.append(cases[-1])

            # Compute the fitness of all individuals in parallel using MPI
            results = concurrent_eval(
                self.fobj, cases, self.comm, allgather=True, model_mpi=self.model_mpi
            )

            # Gather the results
            fit = np.full((self.n_pop,), np.inf)
            for result in results:
                retval, err = result
                if err is not None or retval is None:
                    raise Exception(err)
                else:
                    val, ii = retval
                    fit[ii] = val
        else:
            # Evaluate the population in serial
            fit = np.asarray([self.fobj(ind) for ind in pop])

        # Turn all NaNs in the fitnesses into infs
        fit = np.where(np.isnan(fit), np.inf, fit)

        return fit

    def procreate(self):
        """
        Generate a new population using the selected evolution strategy.

        Returns
        -------
        pop_new : np.array
            Chromosomes of the individuals in the next generation
        f_new : np.array
            New set of mutation rates
        cr_new : np.array
            New set of crossover probabilities
        """
        pop_old_norm = (np.copy(self.pop) - self.lb) / self.range
        pop_new_norm = np.empty_like(pop_old_norm)

        if self.adaptivity == 0 or self.adaptivity == 1:
            if self.adaptivity == 0:
                # No adaptivity. Use static f and cr.
                f_new = self.f
                cr_new = self.cr
            else:
                # Simple adaptivity. Use new f and cr.
                f_new = np.where(
                    self.rng.uniform(size=self.n_pop) < 0.9,
                    self.f,
                    self.rng.uniform(size=self.n_pop) * 0.9 + 1,
                )
                cr_new = np.where(
                    self.rng.uniform(size=self.n_pop) < 0.9,
                    self.cr,
                    self.rng.uniform(size=self.n_pop),
                )

            for idx in range(self.n_pop):
                pop_new_norm[idx], _, _ = self.strategy(
                    idx, pop_old_norm, self.fit, f_new, cr_new, self.rng, False
                )
        else:
            # Complex adaptivity. Mutate f and cr.
            f_new = np.copy(self.f)
            cr_new = np.copy(self.cr)

            for idx in range(self.n_pop):
                pop_new_norm[idx], f_new[idx], cr_new[idx] = self.strategy(
                    idx, pop_old_norm, self.fit, self.f, self.cr, self.rng, True
                )

        pop_new = self.lb + self.range * np.asarray(pop_new_norm)
        return pop_new, f_new, cr_new

    def update(self, pop_new, fit_new, f_new, cr_new):
        """
        Update the population (and f/cr if self-adaptive) and identify the new best and worst individuals.

        Parameters
        ----------
        pop_new : np.array
            Proposed new population resulting from procreation
        fit_new : np.array
            Fitness of the individuals in the new population
        f_new : np.array
            New set of mutation rates
        cr_new : np.array
            New set of crossover probabilities

        Notes
        -----
        Individuals in the old population will only be replaced by the new ones if they have improved fitness.
        Mutation rate and crossover probabilities will only be replaced if self-adaptivity is turned on and if their
        corresponding individuals have improved fitness.
        """
        improved_idxs = np.argwhere(fit_new <= self.fit)
        self.pop[improved_idxs] = pop_new[improved_idxs]
        self.fit[improved_idxs] = fit_new[improved_idxs]

        self.best_idx = np.argmin(self.fit)
        self.best = self.pop[self.best_idx]
        self.best_fit = self.fit[self.best_idx]

        self.worst_idx = np.argmax(self.fit)
        self.worst = self.pop[self.worst_idx]
        self.worst_fit = self.fit[self.worst_idx]

        if self.adaptivity != 0:
            self.f[improved_idxs] = f_new[improved_idxs]
            self.cr[improved_idxs] = cr_new[improved_idxs]
