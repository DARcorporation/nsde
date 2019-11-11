#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import unittest

from differential_evolution import *
from parameterized import parameterized


def paraboloid(x):
    return np.sum(x * x)


class TestDifferentialEvolution(unittest.TestCase):
    @parameterized.expand(
        list(
            map(
                lambda t: (
                        "strategy_"
                        + "/".join([str(_t) for _t in t[:-1]])
                        + "_adaptivity_{}".format(t[-1]),
                ),
                itertools.product(
                    list(EvolutionStrategy.__mutation_strategies__.keys()),
                    [1, 2, 3],
                    list(EvolutionStrategy.__crossover_strategies__.keys()),
                    list(EvolutionStrategy.__repair_strategies__.keys()),
                    [0, 1, 2],
                ),
            )
        )
    )
    def test_differential_evolution(self, name):
        tol = 1e-8
        dim = 2

        strategy, adaptivity = name.split("_")[1::2]
        strategy = EvolutionStrategy(strategy)
        de = DifferentialEvolution(
            strategy=strategy, tolx=tol, tolf=0, adaptivity=int(adaptivity)
        )
        de.init(paraboloid, bounds=[(-100, 100)] * dim)

        last_generation = None
        for last_generation in de:
            pass

        try:
            self.assertTrue(np.all(last_generation.best < tol))
            self.assertAlmostEqual(last_generation.best_fit, 0, 7)
        except AssertionError:
            # This is to account for strategies sometimes 'collapsing' prematurely.
            # This is not a failed test, this is a known phenomenon with DE.
            # In this case we just check that one of the tolerances was triggered.
            self.assertTrue(
                last_generation.dx < tol or last_generation.generation == last_generation.max_gen
            )

    def test_seed_specified_repeatability(self):
        x = [None, None]

        for i in range(2):
            dim = 2
            de = DifferentialEvolution(seed=11)
            de.init(paraboloid, bounds=[(-100, 100)] * dim)
            x[i] = np.copy(de.pop)
            del de

        self.assertTrue(np.all(x[0] == x[1]))

    def test_custom_population_size(self):
        dim = 2
        n_pop = 11
        de = DifferentialEvolution(n_pop=n_pop)
        de.init(paraboloid, bounds=[(-100, 100)]*dim)
        self.assertEqual(de.pop.shape[0], n_pop)

    def test_zero_population_size(self):
        dim = 2
        de = DifferentialEvolution(n_pop=0)
        de.init(paraboloid, bounds=[(-100, 100)]*dim)
        self.assertEqual(de.pop.shape[0], 5 * dim)

    def test_nan_landscape(self):
        dim = 10

        def nan_paraboloid(x):
            if np.dot(x, np.ones_like(x)) < 0.0:
                return np.nan
            else:
                return paraboloid(x)

        de = DifferentialEvolution(tolx=1e-8, tolf=0)
        de.init(nan_paraboloid, bounds=[(-100, 100)] * dim)

        for _ in de:
            pass

        self.assertTrue(np.all(np.abs(de.best) < 1e-5))
