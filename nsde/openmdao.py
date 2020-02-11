#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Driver for the NSDE algorithm.

This driver uses the differential_evolution Python package to provide the logic of the main
differential evolution algorithms, developed by D. de Vries.
"""
import copy
import itertools
import numpy as np
import openmdao
import os

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.mpi import MPI
from six import iteritems, itervalues, next

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(i, _):
        return i


from nsde import NSDE, EvolutionStrategy

if not MPI:
    rank = 0
else:
    rank = MPI.COMM_WORLD.rank


def progress_string(de):
    s = " "
    if tqdm is None:
        s += "gen: {:>5g} / {}, ".format(de.generation, de.max_gen)
    s += (
        "f*: {:> 10.4g}, "
        "dx: {:> 10.4g}, "
        "df: {:> 10.4g}".format(
            de.fit[0, 0],
            de.dx,
            de.df[0] if isinstance(de.df, np.ndarray) else de.df
        )
    )
    if de.n_con > 0:
        s += f", feasible: {np.count_nonzero(np.all(de.con <= 1e-6, axis=1).flatten()):>4d}/{de.n_pop}"
    return s.replace("\n", "")


class NSDEDriver(Driver):
    """
    Driver for the NSDE algorithm.

    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    _de : NSDE
        Differential evolution algorithm.
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since GeneticAlgorithm sees an array of
        design variables.
    _es : EvolutionStrategy
        Evolution strategy to use when evolving the population of the differential evolution algorithm.
    _seed : int
         Seed number which controls the seed and random draws.
    """

    def __init__(self, **kwargs):
        """
        Initialize the NSDE driver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super(NSDEDriver, self).__init__(**kwargs)

        # What we support
        self.supports["inequality_constraints"] = True
        self.supports["equality_constraints"] = True
        self.supports["multiple_objectives"] = True

        # What we don't support yet
        self.supports["integer_design_vars"] = False
        self.supports["two_sided_constraints"] = False
        self.supports["linear_constraints"] = False
        self.supports["simultaneous_derivatives"] = False
        self.supports["active_set"] = False

        self._desvar_idx = {}
        self._es = None
        self._de = None

        # random state can be set for predictability during testing
        if "NSDEDriver_seed" in os.environ:
            self._seed = int(os.environ["NSDEDriver_seed"])
        else:
            self._seed = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            "strategy",
            default="rand-to-best/1/exp/random",
            values=[
                "/".join(strategy)
                for strategy in itertools.product(
                    list(EvolutionStrategy.__mutation_strategies__.keys()),
                    ["1", "2", "3"],
                    list(EvolutionStrategy.__crossover_strategies__.keys()),
                    list(EvolutionStrategy.__repair_strategies__.keys()),
                )
            ],
            desc="Evolution strategy to use for the differential evolution. "
            "An evolution strategy is made up of four parts in fixed order, separated by '/':"
            " mutation strategy ('rand', 'best', or 'rand-to-best'),"
            " number of individuals to involve in the mutation (1, 2, or 3),"
            " crossover strategy ('exp' or 'bin'), and"
            " repair strategy ('random' or 'clip'). "
            "A good introduction of these topics can be found here: "
            "https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/",
        )
        self.options.declare(
            "Pm",
            desc="Mutation rate.",
            default=None,
            lower=0.0,
            upper=1.0,
            allow_none=True,
        )
        self.options.declare(
            "Pc",
            default=None,
            lower=0.0,
            upper=1.0,
            allow_none=True,
            desc="Crossover rate.",
        )
        self.options.declare(
            "adaptivity",
            default=2,
            values=[0, 1, 2],
            desc="Self-adaptivity setting:"
            " 0: mutation and crossover rates are fixed (no self-adaptivity);"
            " 1: mutation and crossover rates are optimized using Monte-Carlo approach; "
            " 2: mutation and crossover rates are optimized using evolutionary algorithm. ",
        )
        self.options.declare(
            "max_gen", default=1000, desc="Number of generations before termination."
        )
        self.options.declare(
            "tolx", default=1e-8, desc="Tolerance of the design vectors' spread."
        )
        self.options.declare(
            "tolf", default=1e-8, desc="Tolerance of the fitness spread."
        )
        self.options.declare(
            "tolc", default=1e-6, desc="Constraint violation tolerance."
        )
        self.options.declare(
            "pop_size",
            default=0,
            desc="Number of individuals (points) to use for the optimization. "
            "If set to 0, it will be calculated automatically as 5 x dimensionality.",
        )
        self.options.declare(
            "run_parallel",
            types=bool,
            default=False,
            desc="Set to True to execute the points in a generation in parallel.",
        )
        self.options.declare(
            "procs_per_model",
            default=1,
            lower=1,
            desc="Number of processors to give each model under MPI.",
        )
        self.options.declare(
            "show_progress",
            default=False,
            desc="Set to true if a progress bar should be shown.",
        )
        self.options.declare(
            "generation_callback",
            default=None,
            allow_none=True,
            desc="Callback which will be called for each generation."
            "Callable should have a single argument, which will "
            "be an instance of the NSDE class.",
        )
        self.options.declare(
            "initial_population",
            default=None,
            allow_none=True,
            desc="Initial population with which to start the optimization."
                 "This should be a 2D array with design vectors as rows and"
                 "one row per individual.",
        )

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(NSDEDriver, self)._setup_driver(problem)

        model_mpi = None
        comm = self._problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options["run_parallel"]:
            comm = None

        self._es = EvolutionStrategy(self.options["strategy"])
        self._de = NSDE(
            strategy=self._es,
            mut=self.options["Pm"],
            crossp=self.options["Pc"],
            adaptivity=self.options["adaptivity"],
            max_gen=self.options["max_gen"],
            tolx=self.options["tolx"],
            tolf=self.options["tolf"],
            tolc=self.options["tolc"],
            n_pop=self.options["pop_size"],
            seed=self._seed,
            comm=comm,
            model_mpi=model_mpi,
        )

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Here, we generate the model communicators.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        procs_per_model = self.options["procs_per_model"]
        if MPI and self.options["run_parallel"]:

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError(
                    "The total number of processors is not evenly divisible by the "
                    "specified number of processors per model.\n Provide a "
                    "number of processors that is a multiple of %d, or "
                    "specify a number of processors per model that divides "
                    "into %d." % (procs_per_model, full_size)
                )
            color = comm.rank % size
            model_comm = comm.Split(color)

            # Everything we need to figure out which case to run.
            self._concurrent_pop_size = size
            self._concurrent_color = color

            return model_comm

        self._concurrent_pop_size = 0
        self._concurrent_color = 0
        return comm

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "NSDE"

    def get_de(self):
        """
        Get a copy of the driver's underlying DE class.

        Returns
        -------
        NSDE
            A copy of the driver's underlying DE class
        """
        return copy.copy(self._de)

    def run(self):
        """
        Execute the differential evolution algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem.model
        de = self._de

        de.strategy = EvolutionStrategy(self.options["strategy"])
        de.f = self.options["Pm"]
        de.cr = self.options["Pc"]
        de.adaptivity = self.options["adaptivity"]
        de.n_pop = self.options["pop_size"]
        de.max_gen = self.options["max_gen"]
        de.tolx = self.options["tolx"]
        de.tolf = self.options["tolf"]
        de.tolc = self.options["tolc"]

        self._check_for_missing_objective()

        # Size design variables.
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()

        count = 0
        for name, meta in iteritems(desvars):
            if name in self._designvars_discrete:
                val = desvar_vals[name]
                if np.isscalar(val):
                    size = 1
                else:
                    size = len(val)
            else:
                size = meta["size"]
            self._desvar_idx[name] = (count, count + size)
            count += size

        bounds = []
        x0 = np.empty(count)

        # Figure out bounds vectors and initial design vars
        for name, meta in iteritems(desvars):
            i, j = self._desvar_idx[name]
            lb = meta["lower"]
            if isinstance(lb, float):
                lb = [lb] * (j - i)
            ub = meta["upper"]
            if isinstance(ub, float):
                ub = [ub] * (j - i)
            for k in range(j - i):
                bounds += [(lb[k], ub[k])]
            x0[i:j] = desvar_vals[name]

        de.init(self.objective_callback, bounds, self.options["initial_population"])
        if rank == 0 and self.options["show_progress"]:
            print(progress_string(de))
        if self.options["generation_callback"] is not None:
            self.options["generation_callback"](de)

        gen_iter = de
        if rank == 0 and self.options["show_progress"] and tqdm is not None:
            gen_iter = tqdm(gen_iter, total=self.options["max_gen"])

        for generation in gen_iter:
            if rank == 0 and self.options["show_progress"]:
                print(progress_string(generation))
                if self.options["generation_callback"] is not None:
                    self.options["generation_callback"](de)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        best = de.pop[0]
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = best[i:j]
            self.set_design_var(name, val)

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False

    def objective_callback(self, x):
        r"""
        Evaluate problem objective at the requested point.

        Parameters
        ----------
        x : ndarray
            Value of design variables.

        Returns
        -------
        f : ndarray
            Objective values
        g : ndarray
            Constraint values
        """
        model = self._problem.model

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()

            # Get the objective functions' values
            f = np.array(list(itervalues(self.get_objective_values())))

            # Get the constraint violations
            g = np.array([])
            with np.errstate(
                divide="ignore"
            ):  # Ignore divide-by-zero warnings temporarily
                for name, val in iteritems(self.get_constraint_values()):
                    con = self._cons[name]
                    # The not used fields will either None or a very large number
                    # All constraints will be converted into standard <= 0 form.
                    if (con["lower"] is not None) and np.any(
                        con["lower"] > -almost_inf
                    ):
                        g = np.append(
                            g,
                            np.where(
                                con["lower"] == 0,
                                con["lower"] - val,
                                1 - val / con["lower"],
                            ).flatten(),
                        )
                    elif (con["upper"] is not None) and np.any(
                        con["upper"] < almost_inf
                    ):
                        g = np.append(
                            g,
                            np.where(
                                con["upper"] == 0,
                                val - con["upper"],
                                val / con["upper"] - 1,
                            ).flatten(),
                        )
                    elif (con["equals"] is not None) and np.any(
                        np.abs(con["equals"]) < almost_inf
                    ):
                        g = np.append(
                            g,
                            np.where(
                                con["equals"] == 0,
                                np.abs(con["equals"] - val),
                                np.abs(1 - val / con["equals"]),
                            ).flatten()
                            - self.options["tolc"],
                        )

            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        return f.flatten(), g.flatten()
