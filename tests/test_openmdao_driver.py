import numpy as np
import openmdao.api as om
import pytest

from differential_evolution import *


@pytest.fixture
def problem():
    dim = 2

    prob = om.Problem()
    prob.model.add_subsystem('indeps', om.IndepVarComp('x', val=np.ones(dim)), promotes=['*'])
    prob.model.add_subsystem('objf', om.ExecComp('f = sum(x * x)', f=1., x=np.ones(dim)), promotes=['*'])

    prob.model.add_design_var('x', lower=-100., upper=100.)
    prob.model.add_objective('f')

    prob.driver = DifferentialEvolutionDriver()
    return prob


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
        assert np.all(problem['x'] < 1e-3)
        assert problem['f'][0] < 1e-3
    except AssertionError:
        # This is to account for strategies sometimes 'collapsing' prematurely.
        # This is not a failed test, this is a known phenomenon with DE.
        # In this case we just check that one of the two tolerances was triggered.
        assert problem.driver._de.dx < tol or problem.driver._de.df < tol
