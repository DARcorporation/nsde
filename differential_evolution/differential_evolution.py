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
        Mutation rate and crossover probability
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

    def __init__(self, strategy=None, mut=0.85, crossp=1.,
                 max_gen=100, tolx=1e-6, tolf=1e-6,
                 n_pop=None, seed=None, comm=None, model_mpi=None):
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

        self.comm = comm
        self.model_mpi = model_mpi

        self.strategy = strategy
        if strategy is None:
            self.strategy = EvolutionStrategy("best-to-rand/1/exp")

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
        if self.f is None or 0. > self.f > 1.:
            self.f = 0.85
        if self.cr is None or 0. > self.cr > 1.:
            self.cr = 1.

        self.fobj = fobj if self.comm is None else mpi_fobj_wrapper(fobj)
        self.lb, self.ub = np.asarray(bounds).T
        self.range = self.ub - self.lb

        self.n_dim = len(bounds)

        if pop is not None:
            self.n_pop = pop.shape[0]
            self.pop = pop
        else:
            self.n_pop = self.n_dim * 5
            self.pop = self.rng.uniform(self.lb, self.ub, size=(self.n_pop, self.n_dim))

        self.fit = self(self.pop)
        self.update(self.pop, self.fit)

        self.generation = 0

        self._is_initialized = True

    @property
    def is_initialized(self):
        """bool: True if the algorithm has been initialized, False if not."""
        return self._is_initialized

    def __iter__(self):
        """
        Main evolution loop

        Yields
        ------
        self
            A copy of this class at each generation
        """
        if not self._is_initialized:
            raise RuntimeError("DifferentialEvolution is not yet initialized.")

        while self.generation < self.max_gen:
            pop_new = self.procreate()
            fit_new = self(pop_new)
            self.update(pop_new, fit_new)

            self.dx = np.sum((self.range * (self.worst - self.best)) ** 2) ** 0.5
            self.df = np.abs(self.worst_fit - self.best_fit)
            self.generation += 1

            yield self

            if self.dx < self.tolx:
                break
            if self.df < self.tolf:
                break

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
            # Use population of rank 0 on all processors
            pop = self.comm.bcast(pop, root=0)

            cases = [((item, ii), None) for ii, item in enumerate(pop)]
            # Pad the cases with some dummy cases to make the cases divisible amongst the procs.
            extra = len(cases) % self.comm.size
            if extra > 0:
                for j in range(self.comm.size - extra):
                    cases.append(cases[-1])

            results = concurrent_eval(self.fobj, cases, self.comm, allgather=True,
                                      model_mpi=self.model_mpi)

            fit = np.full((self.n_pop,), np.inf)
            for result in results:
                (val, ii), _ = result
                fit[ii] = val
        else:
            fit = [self.fobj(ind) for ind in pop]
        return np.asarray(fit)

    def procreate(self):
        """
        Generate a new population using the selected evolution strategy.

        Returns
        -------
        np.array
            Chromosomes of the individuals in the next generation
        """
        pop_old_norm = (np.copy(self.pop) - self.lb) / self.range
        pop_new_norm = [self.strategy(idx, pop_old_norm, self.fit, self.f, self.cr, self.rng) for idx in range(self.n_pop)]
        return self.lb + self.range * np.asarray(pop_new_norm)

    def update(self, pop_new, fit_new):
        """
        Update the population and identify the new best and worst individuals.

        Parameters
        ----------
        pop_new : np.array
            Proposed new population resulting from procreation
        fit_new : np.array
            Fitness of the individuals in the new population

        Notes
        -----
        Individuals in the old population will only be replaced by the new ones if they have improved fitness.
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
