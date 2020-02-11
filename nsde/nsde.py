#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Definition of the NSDE algorithm"""
import numpy as np

try:
    from openmdao.utils.concurrent import concurrent_eval
except ModuleNotFoundError:
    import warnings

    warnings.warn("OpenMDAO is not installed. Concurrent evaluation is not available.")

from . import sorting
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


class NSDE:
    """
    Non-dominated Sorting Differential Evolution (NSDE) Algorithm.

    Attributes
    ----------
    fobj : callable
        Objective function.
        Should have a single argument of type array_like which corresponds to the design vector.
        Should have either a single float or 1D array output corresponding to the objective function value(s),
        or two array_like outputs, the first of which corresponds to the objective function value(s) and the second
        to the constraint violations.
        Constraints are assumed to be satisfied if constraint violations <= constraint tolerance.
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
    tolc : float
        Constraint violation tolerance.
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
    con : np.array
        Constraint violations of the individuals in the population
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
        tolc=1e-6,
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
        self.tolc = tolc

        self.n_dim = 0
        self.n_obj = 0
        self.n_con = 0
        self.n_pop = n_pop

        self.rng = np.random.default_rng(seed)

        if adaptivity not in [0, 1, 2]:
            raise ValueError("self_adaptivity must be one of (0, 1, 2).")
        self.adaptivity = adaptivity

        self.comm = comm
        self.model_mpi = model_mpi

        if strategy is None:
            self.strategy = EvolutionStrategy("rand-to-best/1/bin/random")
        elif isinstance(strategy, EvolutionStrategy):
            self.strategy = strategy
        elif isinstance(strategy, str):
            self.strategy = EvolutionStrategy(strategy)
        else:
            raise ValueError(
                "Argument `strategy` should be None, a str, or an instance of EvolutionStrategy."
            )

        self.pop = None
        self.fit = None
        self.con = None
        self.fronts = None
        self.dx, self.df = np.inf, np.inf

        self.generation = 0

        self._is_initialized = False
        self._running_under_mpi = comm is not None and hasattr(comm, "bcast")

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

        def create_f_cr(adaptivity, f, cr, n, rng):
            # Create random mutation/crossover parameters if self-adaptivity is used
            if adaptivity == 0:
                f = f * np.ones(n)
                cr = cr * np.ones(n)
            elif adaptivity == 1:
                f = rng.uniform(size=n) * 0.9 + 0.1
                cr = rng.uniform(size=n)
            elif adaptivity == 2:
                f = rng.uniform(size=n) * 0.15 + 0.5
                cr = rng.uniform(size=n) * 0.15 + 0.5
            return f, cr

        adjust_pop = False
        if pop is not None:
            self.n_pop = pop.shape[0]
            self.pop = pop
            self.f, self.cr = create_f_cr(
                self.adaptivity, self.f, self.cr, self.n_pop, self.rng
            )
        else:
            if self.n_pop is None or self.n_pop <= 0:
                self.pop = self.rng.uniform(self.lb, self.ub, size=(1, self.n_dim))
                adjust_pop = True
                self.n_pop = 1
            else:
                self.pop = self.rng.uniform(
                    self.lb, self.ub, size=(self.n_pop, self.n_dim)
                )
                self.f, self.cr = create_f_cr(
                    self.adaptivity, self.f, self.cr, self.n_pop, self.rng
                )

        # Ensure all processors have the same population and mutation/crossover parameters
        if self._running_under_mpi:
            self.pop, self.f, self.cr = self.comm.bcast(
                (self.pop, self.f, self.cr), root=0
            )

        self.fit, self.con = self(self.pop)

        self.n_obj = self.fit.shape[1]
        if self.con is not None:
            self.n_con = self.con.shape[1]

        if adjust_pop:
            self.n_pop = 5 * self.n_dim * self.n_obj

            # If we are running under MPI, expand population to fully exploit all processors
            if self._running_under_mpi:
                self.n_pop = int(np.ceil(self.n_pop / self.comm.size) * self.comm.size)

            self.pop = np.concatenate(
                (
                    self.pop,
                    self.rng.uniform(
                        self.lb, self.ub, size=(self.n_pop - 1, self.n_dim)
                    ),
                )
            )
            self.f, self.cr = create_f_cr(
                self.adaptivity, self.f, self.cr, self.n_pop, self.rng
            )

            if self._running_under_mpi:
                self.pop, self.f, self.cr = self.comm.bcast(
                    (self.pop, self.f, self.cr), root=0
                )

            self.fit, self.con = self(self.pop)

        self.update()

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
            raise RuntimeError("NSDE is not yet initialized.")
        return self

    def __next__(self):
        """
        Main iteration.

        Returns
        -------
        NSDE
            The new state at the next generation.
        """
        if (
            self.generation < self.max_gen
            and self.dx > self.tolx
            and self.df > self.tolf
        ):
            # Create a new population and mutation/crossover parameters
            pop_new, f_new, cr_new = self.procreate()

            # Ensure all processors have the same updated population and mutation/crossover parameters
            if self._running_under_mpi:
                pop_new, f_new, cr_new = self.comm.bcast(
                    (pop_new, f_new, cr_new), root=0
                )

            # Evaluate the fitness of the new population
            fit_new, con_new = self(pop_new)

            # Update the class with the new data
            self.update(pop_new, fit_new, con_new, f_new, cr_new)

            # Compute spreads and update generation counter
            if self.n_obj == 1:
                self.dx = np.linalg.norm(self.pop[0] - self.pop[-1])
                self.df = np.abs(self.fit[0] - self.fit[-1])
            else:
                # TODO: Find a way to measure convergence for pareto based optimization
                self.dx = np.inf
                self.df = np.inf
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
        fit : np.array
            Fitness of the individuals in the given population
        con : np.array or None
            Constraint violations of the individuals in the given population if present. None otherwise.

        Notes
        -----
        If this class has an MPI communicator the individuals will be evaluated in parallel.
        Otherwise function evaluation will be serial.
        """
        if self.is_initialized:
            fit = np.empty((self.n_pop, self.n_obj))
            con = None if self.n_con is None else np.empty((self.n_pop, self.n_con))
        else:
            fit = pop.shape[0] * [None]
            con = None

        def handle_result(_v, _i, _fit, _con):
            if isinstance(_v, tuple):
                _fit[_i] = np.asarray(_v[0])
                c = np.asarray(_v[1])
                if _con is None:
                    _con = np.empty((pop.shape[0], c.size))
                _con[_i] = c
            else:
                _fit[_i] = _v
            return _fit, _con

        # Evaluate generation
        if self._running_under_mpi:
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
            for result in results:
                retval, err = result
                if err is not None or retval is None:
                    raise Exception(err)
                else:
                    fit, con = handle_result(*retval, fit, con)
        else:
            # Evaluate the population in serial
            for idx, ind in enumerate(pop):
                val = self.fobj(ind)
                fit, con = handle_result(val, idx, fit, con)

        # Turn all NaNs in the fitnesses into infs
        fit = np.reshape(np.where(np.isnan(fit), np.inf, fit), (pop.shape[0], -1))
        if con is not None:
            con = np.reshape(np.where(np.isnan(con), np.inf, con), (pop.shape[0], -1))
        return fit, con

    def run(self):
        for _ in self:
            pass

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

        # If there are constraints, augment the fitness to penalize infeasible individuals while procreating.
        # This stops the best and rand-to-best strategies to keep the best infeasible individual alive indefinitely.
        if self.n_con and False:
            fit = np.where(
                np.any(self.con >= 1e-6, axis=1, keepdims=True),
                np.linalg.norm(self.con, axis=1, keepdims=True) + np.max(self.fit),
                self.fit,
            )
        else:
            fit = self.fit

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
                    idx, pop_old_norm, fit, self.fronts, f_new, cr_new, self.rng, False
                )
        else:
            # Complex adaptivity. Mutate f and cr.
            f_new = np.copy(self.f)
            cr_new = np.copy(self.cr)

            for idx in range(self.n_pop):
                pop_new_norm[idx], f_new[idx], cr_new[idx] = self.strategy(
                    idx, pop_old_norm, fit, self.fronts, self.f, self.cr, self.rng, True
                )

        pop_new = self.lb + self.range * np.asarray(pop_new_norm)
        return pop_new, f_new, cr_new

    def update(self, pop_new=None, fit_new=None, con_new=None, f_new=None, cr_new=None):
        """
        Update the population (and f/cr if self-adaptive).

        Parameters
        ----------
        pop_new : np.array or None, optional
            Proposed new population resulting from procreation
        fit_new : np.array or None, optional
            Fitness of the individuals in the new population
        con_new : np.array or None, optional
            Constraint violations of the individuals in the new population
        f_new : np.array or None, optional
            New set of mutation rates
        cr_new : np.array or None, optional
            New set of crossover probabilities

        Notes
        -----
        Individuals in the old population will only be replaced by the new ones if they have improved fitness.
        Mutation rate and crossover probabilities will only be replaced if self-adaptivity is turned on and if their
        corresponding individuals have improved fitness.
        """
        if self.n_obj == 1:
            self._update_single(pop_new, fit_new, con_new, f_new, cr_new)
        else:
            self._update_multi(pop_new, fit_new, con_new, f_new, cr_new)

    def _update_single(
        self, pop_new=None, fit_new=None, con_new=None, f_new=None, cr_new=None
    ):
        if self.n_con:
            cs = np.sum(
                np.where(np.greater(self.con, self.tolc), self.con, 0.0),
                axis=1
            )
        else:
            cs = 0

        if (
            pop_new is not None
            and fit_new is not None
            and f_new is not None
            and cr_new is not None
        ):
            if self.n_con:
                c_new = np.all(con_new <= self.tolc, axis=1)
                c_old = np.all(self.con <= self.tolc, axis=1)
                cs_new = np.sum(
                    np.where(np.greater(con_new, self.tolc), con_new, 0.0), axis=1
                )

                improved_indices = np.argwhere(
                    ((c_new & c_old) & (fit_new <= self.fit).flatten())
                    + (c_new & ~c_old)
                    + ((~c_new & ~c_old) & (cs_new <= cs))
                )

                self.con[improved_indices] = con_new[improved_indices]
                cs[improved_indices] = cs_new[improved_indices]
            else:
                improved_indices = np.argwhere((fit_new <= self.fit).flatten())

            self.pop[improved_indices] = pop_new[improved_indices]
            self.fit[improved_indices] = fit_new[improved_indices]

            if self.adaptivity != 0:
                self.f[improved_indices] = f_new[improved_indices]
                self.cr[improved_indices] = cr_new[improved_indices]

        # Sort population so the best individual is always the first
        idx_sort = np.argsort(
            self.fit.flatten() + np.where(cs != 0.0, cs * np.max(self.fit), 0.0)
        )

        self.pop = self.pop[idx_sort]
        self.fit = self.fit[idx_sort]
        if self.n_con:
            self.con = self.con[idx_sort]

        if self.adaptivity != 0:
            self.f = self.f[idx_sort]
            self.cr = self.cr[idx_sort]

    def _update_multi(
        self, pop_new=None, fit_new=None, con_new=None, f_new=None, cr_new=None
    ):
        if (
            pop_new is not None
            and fit_new is not None
            and f_new is not None
            and cr_new is not None
        ):
            self.pop = np.concatenate((self.pop, pop_new))
            self.fit = np.concatenate((self.fit, fit_new))
            if self.n_con:
                self.con = np.concatenate((self.con, con_new))

            if self.adaptivity != 0:
                self.f = np.concatenate((self.f, f_new))
                self.cr = np.concatenate((self.cr, cr_new))

        if self.n_con:
            fronts = sorting.nonDominatedSorting(self.fit, self.con, self.n_pop)
        else:
            fronts = sorting.nonDominatedSorting(self.fit, self.n_pop)
        fronts[-1] = np.asarray(fronts[-1])[
            sorting.crowdingDistanceSorting(self.fit[fronts[-1]])[
                : (self.n_pop - sum(len(f) for f in fronts[:-1]))
            ]
        ].tolist()
        new_idxs = []
        for front in fronts:
            new_idxs += front

        self.fronts = [list(range(i, i + len(f))) for i, f in enumerate(fronts)]

        self.pop = self.pop[new_idxs]
        self.fit = self.fit[new_idxs]
        if self.n_con:
            self.con = self.con[new_idxs]

        if self.adaptivity != 0:
            self.f = self.f[new_idxs]
            self.cr = self.cr[new_idxs]
