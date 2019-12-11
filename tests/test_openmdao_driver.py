#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Unit tests for the DifferentialEvolution Driver."""
import itertools
import numpy as np
import openmdao.api as om
import os
import unittest

from parameterized import parameterized

from nsde import *

all_strategies = list(
            map(
                lambda t: (
                    "strategy_"
                    + "/".join([str(_t) for _t in t[:-1]])
                    + "_adaptivity_{}".format(t[-1]),
                ),
                itertools.product(
                    list(EvolutionStrategy.__mutation_strategies__.keys()),
                    [1, 2],
                    list(EvolutionStrategy.__crossover_strategies__.keys()),
                    list(EvolutionStrategy.__repair_strategies__.keys()),
                    [0, 1, 2],
                ),
            )
        )


class TestSingleObjective(unittest.TestCase):
    def setUp(self):
        os.environ["DifferentialEvolutionDriver_seed"] = "11"
        self.dim = 2

        prob = om.Problem()
        prob.model.add_subsystem(
            "indeps", om.IndepVarComp("x", val=np.ones(self.dim)), promotes=["*"]
        )
        prob.model.add_subsystem(
            "objf",
            om.ExecComp("f = sum(x * x)", f=1.0, x=np.ones(self.dim)),
            promotes=["*"],
        )

        prob.model.add_design_var("x", lower=-100.0, upper=100.0)
        prob.model.add_objective("f")

        prob.driver = DifferentialEvolutionDriver()
        self.problem = prob

    def tearDown(self):
        self.problem.cleanup()

    @parameterized.expand(all_strategies)
    def test_unconstrained(self, name):
        tol = 1e-8

        strategy, adaptivity = name.split("_")[1::2]
        self.problem.driver.options["strategy"] = strategy
        self.problem.driver.options["adaptivity"] = int(adaptivity)
        self.problem.setup()
        self.problem.run_driver()

        for x in self.problem["x"]:
            self.assertAlmostEqual(x, 0.0, 1)
        self.assertAlmostEqual(self.problem["f"][0], 0.0, 2)

    def test_constrained(self):
        f_con = om.ExecComp("c = 1 - x[0]", c=0.0, x=np.ones(self.dim))
        self.problem.model.add_subsystem("con", f_con, promotes=["*"])
        self.problem.model.add_constraint("c", upper=0.0)

        self.problem.setup()
        self.problem.run_driver()

        self.assertAlmostEqual(self.problem["x"][0], 1.0, 1)
        for x in self.problem["x"][1:]:
            self.assertAlmostEqual(x, 0, 1)
        self.assertAlmostEqual(self.problem["f"][0], 1.0, 2)

    def test_vectorized_constraints(self):
        self.problem.model.add_subsystem(
            "con",
            om.ExecComp("c = 1 - x", c=np.zeros(self.dim), x=np.ones(self.dim)),
            promotes=["*"],
        )
        self.problem.model.add_constraint("c", upper=np.zeros(self.dim))

        self.problem.setup()
        self.problem.run_driver()

        for x in self.problem["x"]:
            self.assertAlmostEqual(x, 1.0, 2)
        self.assertAlmostEqual(self.problem["f"][0], self.dim, 2)

    def test_seed_specified_repeatability(self):
        x = [None, None]
        f = [None, None]

        for i in range(2):
            self.assertEqual(self.problem.driver._seed, 11)

            self.problem.driver.options["max_gen"] = 10
            self.problem.setup()
            self.problem.run_driver()

            x[i] = self.problem["x"]
            f[i] = self.problem["f"][0]

            self.tearDown()
            self.setUp()

        self.assertTrue(np.all(x[0] == x[1]))
        self.assertEqual(f[0], f[1])

    def test_custom_population_size(self):
        n_pop = 11
        self.problem.driver.options["pop_size"] = n_pop
        self.problem.setup()
        self.problem.run_driver()
        self.assertEqual(self.problem.driver._de.n_pop, n_pop)
        self.assertEqual(self.problem.driver._de.pop.shape[0], n_pop)


class TestMultiObjective(unittest.TestCase):

    def test_unconstrained(self):
        # The test problem is the multi-objective Schaffer Problem No.1
        prob = om.Problem()
        prob.model.add_subsystem(
            "indeps", om.IndepVarComp("x", val=1.0), promotes=["*"]
        )
        prob.model.add_subsystem(
            "objf",
            om.ExecComp("f = [x[0] ** 2, (x[0] - 2) ** 2]", f=[1.0, 1.0], x=1.0),
            promotes=["*"],
        )

        prob.model.add_design_var("x", lower=-100.0, upper=100.0)
        prob.model.add_objective("f")

        prob.driver = DifferentialEvolutionDriver()
        self.problem = prob
        self.problem.setup()
        self.problem.run_driver()

        # Check that the solution is on the known pareto front
        self.assertAlmostEqual(self.problem["f"][1], (self.problem["f"][0] ** 0.5 - 2) ** 2, 2)

        self.problem.cleanup()


if __name__ == "__main__":
    unittest.main()
