import numpy as np
import openmdao.api as om

from differential_evolution import DifferentialEvolutionDriver


def problem(dim):
    prob = om.Problem()
    prob.model.add_subsystem('indeps', om.IndepVarComp('x', val=np.ones(dim)), promotes=['*'])
    prob.model.add_subsystem('objf', om.ExecComp('f = sum(x * x)', f=1., x=np.ones(dim)), promotes=['*'])

    prob.model.add_design_var('x', lower=-100., upper=100.)
    prob.model.add_objective('f')

    prob.driver = DifferentialEvolutionDriver()
    return prob


def test_openmdao_driver():
    prob = problem(2)
    prob.setup()
    prob.run_driver()

    assert np.all(prob['x'] < 1e-4)
    assert prob['f'][0] < 1e-4
