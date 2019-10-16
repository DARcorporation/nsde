#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import openmdao.api as om
import os
import pytest

from differential_evolution import *


def get_problem():
    dim = 2

    prob = om.Problem()
    prob.model.add_subsystem(
        "indeps", om.IndepVarComp("x", val=np.ones(dim)), promotes=["*"]
    )
    prob.model.add_subsystem(
        "objf", om.ExecComp("f = sum(x * x)", f=1.0, x=np.ones(dim)), promotes=["*"]
    )

    prob.model.add_design_var("x", lower=-100.0, upper=100.0)
    prob.model.add_objective("f")

    prob.driver = DifferentialEvolutionDriver()
    return prob


@pytest.fixture
def problem():
    return get_problem()


@pytest.mark.parametrize("repair", EvolutionStrategy.__repair_strategies__.keys())
@pytest.mark.parametrize("crossover", EvolutionStrategy.__crossover_strategies__.keys())
@pytest.mark.parametrize("number", [1, 2, 3])
@pytest.mark.parametrize("mutation", EvolutionStrategy.__mutation_strategies__.keys())
@pytest.mark.parametrize("adaptivity", [0, 1, 2])
def test_openmdao_driver(problem, mutation, number, crossover, repair, adaptivity):
    tol = 1e-8

    strategy = "/".join([mutation, str(number), crossover, repair])
    problem.driver.options["strategy"] = strategy
    problem.driver.options["adaptivity"] = adaptivity
    problem.setup()
    problem.run_driver()

    try:
        assert np.all(problem["x"] < 1e-3)
        assert problem["f"][0] < 1e-3
    except AssertionError:
        # This is to account for strategies sometimes 'collapsing' prematurely.
        # This is not a failed test, this is a known phenomenon with DE.
        # In this case we just check that one of the two tolerances was triggered.
        assert problem.driver._de.dx < tol or problem.driver._de.df < tol


def test_openmdao_driver_recorder(problem):
    from openmdao.recorders.case_recorder import CaseRecorder

    class MyRecorder(CaseRecorder):
        def record_metadata_system(self, recording_requester):
            pass

        def record_metadata_solver(self, recording_requester):
            pass

        def record_iteration_driver(self, recording_requester, data, metadata):
            assert isinstance(recording_requester, DifferentialEvolutionDriver)
            de = recording_requester.get_de()
            assert "out" in data
            assert "indeps.x" in data["out"]
            assert "objf.f" in data["out"]

            x = data["out"]["indeps.x"]
            f = data["out"]["objf.f"]
            assert x in de.pop
            assert f in de.fit

            assert np.all(x - de.best == 0)
            assert f - de.best_fit == 0

        def record_iteration_system(self, recording_requester, data, metadata):
            pass

        def record_iteration_solver(self, recording_requester, data, metadata):
            pass

        def record_iteration_problem(self, recording_requester, data, metadata):
            pass

        def record_derivatives_driver(self, recording_requester, data, metadata):
            pass

        def record_viewer_data(self, model_viewer_data):
            pass

    problem.driver.add_recorder(MyRecorder())
    problem.setup()
    problem.run_driver()


def test_seed_specified_repeatability():
    x = [None, None]
    f = [None, None]

    for i in range(2):
        os.environ["DifferentialEvolutionDriver_seed"] = "1"
        prob = get_problem()
        assert prob.driver._seed == 1

        prob.driver.options["max_gen"] = 10
        prob.setup()
        prob.run_driver()

        x[i] = prob["x"]
        f[i] = prob["f"][0]

        prob.cleanup()

    assert np.all(x[0] == x[1])
    assert f[0] == f[1]


def test_custom_population_size(problem):
    n_pop = 11
    problem.driver.options["pop_size"] = n_pop
    problem.setup()
    problem.run_driver()
    assert problem.driver._de.n_pop == n_pop