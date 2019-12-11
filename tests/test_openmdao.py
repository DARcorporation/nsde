#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" Unit tests for the NSDE Driver."""
import openmdao.api as om
import os
import pytest

from nsde import NSDEDriver
from . import all_strategies, get_strategy_designation


def get_problem(dim=2):
    os.environ["NSDEDriver_seed"] = "11"

    prob = om.Problem()
    prob.model.add_subsystem(
        "indeps", om.IndepVarComp("x", val=np.ones(dim)), promotes=["*"]
    )
    prob.model.add_subsystem(
        "objf",
        om.ExecComp("f = sum(x * x)", f=1.0, x=np.ones(dim)),
        promotes=["*"],
    )

    prob.model.add_design_var("x", lower=-100.0, upper=100.0)
    prob.model.add_objective("f")

    prob.driver = NSDEDriver()
    return prob


@pytest.fixture
def problem():
    dim = 2
    prob = get_problem(dim)
    yield prob, dim
    prob.cleanup()


def _go_problem(prob, *args):
    prob.driver.options["strategy"] = get_strategy_designation(*args[:-1])
    prob.driver.options["adaptivity"] = args[-1]
    problem.setup()
    problem.run_driver()


@all_strategies
def test_single_objective_unconstrained(problem, *args):
    problem, dim = problem
    _go_problem(problem, *args)

    assert problem["x"] == pytest.approx(0.0, abs=1e-1)
    assert problem["f"] == pytest.approx(0.0, abs=1e-1)


@all_strategies
def test_single_objective_constrained(problem, *args):
    problem, dim = problem

    f_con = om.ExecComp("c = 1 - x[0]", c=0.0, x=np.ones(dim))
    problem.model.add_subsystem("con", f_con, promotes=["*"])
    problem.model.add_constraint("c", upper=0.0)

    _go_problem(problem, *args)

    assert problem["x"][0] == pytest.approx(1.0, abs=1e-1)
    assert problem["x"][1:] == pytest.approx(0.0, abs=1e-1)
    assert problem["f"] == pytest.approx(1.0, abs=1e-2)


@all_strategies
def test_single_objective_vectorized_constraints(problem, *args):
    problem, dim = problem

    problem.model.add_subsystem(
            "con",
            om.ExecComp("c = 1 - x", c=np.zeros(dim), x=np.ones(dim)),
            promotes=["*"],
        )
    problem.model.add_constraint("c", upper=np.zeros(dim))

    _go_problem(problem, *args)

    assert problem["x"] == pytest.approx(1.0, abs=1e-1)
    assert problem["f"] == pytest.approx(dim, abs=1e-1)


def test_seed_specified_repeatability():
    for i in range(2):
        prob = get_problem()
        assert prob.driver._seed == 11

        prob.driver.options["max_gen"] = 10
        prob.setup()
        prob.run_driver()

        if i == 0:
            x = prob["x"]
            f = prob["f"][0]
        else:
            assert np.all(x == prob["x"])
            assert f == prob["f"][0]

        prob.cleanup()


def test_custom_population_size(problem):
    problem, dim = problem

    n_pop = 11
    problem.driver.options["pop_size"] = n_pop
    problem.setup()
    problem.run_driver()

    de = problem.driver.get_de()
    assert de.n_pop == n_pop
    assert de.pop.shape[0] == n_pop


def test_multi_objective_unconstrainted():
    # The test problem is the multi-objective Schaffer Problem No.1
    problem = om.Problem()
    problem.model.add_subsystem(
        "indeps", om.IndepVarComp("x", val=1.0), promotes=["*"]
    )
    problem.model.add_subsystem(
        "objf",
        om.ExecComp("f = [x[0] ** 2, (x[0] - 2) ** 2]", f=[1.0, 1.0], x=1.0),
        promotes=["*"],
    )

    problem.model.add_design_var("x", lower=-100.0, upper=100.0)
    problem.model.add_objective("f")

    problem.driver = NSDEDriver()
    problem.setup()
    problem.run_driver()

    # Check that the solution is on the known pareto front
    assert problem["f"][1] == pytest.approx((problem["f"][0] ** 0.5 - 2) ** 2, abs=1e-2)

    problem.cleanup()
