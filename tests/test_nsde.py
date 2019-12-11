#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import unittest

from nsde import *
from parameterized import parameterized


def paraboloid(x):
    return np.sum(x * x)


def schaffer_n1(x):
    return [x ** 2, (x - 2) ** 2]


def binh_and_korn(x):
    return (
        [4 * np.sum(x ** 2), np.sum((x - 5) ** 2)],
        [((x[0] - 5) ** 2 + x[1] ** 2) / 25 - 1, 1 - ((x[0] - 8) ** 2 + (x[1] + 3) ** 2) / 7.7]
    )

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

    def _test(self, fobj, x_opt, f_opt, name):
        strategy, adaptivity = name.split("_")[1::2]
        strategy = EvolutionStrategy(strategy)
        de = NSDE(
            strategy=strategy, tolf=0, adaptivity=int(adaptivity)
        )
        de.init(fobj, bounds=[(-100, 100)] * 2)

        for _ in de:
            pass

        x_close = np.all(np.abs(de.best - x_opt) < 1e-2)
        if not x_close and adaptivity == 0 or adaptivity == 1:
            self.assertTrue(de.dx <= de.tolx or de.generation >= de.max_gen)
        else:
            self.assertTrue(x_close)
            self.assertAlmostEqual(de.best_fit[0], f_opt, 2)

    @parameterized.expand(all_strategies)
    def test_unconstrained(self, name):
        self._test(paraboloid, 0, 0, name)

    @parameterized.expand(all_strategies)
    def test_constrained(self, name):
        def fobj(x):
            return paraboloid(x), 1 - x
        self._test(fobj, 1, 2, name)

    def test_nan_landscape(self):
        dim = 10

        def nan_paraboloid(x):
            if np.dot(x, np.ones_like(x)) < 0.0:
                return np.nan
            else:
                return paraboloid(x)

        de = NSDE(tolx=1e-8, tolf=0)
        de.init(nan_paraboloid, bounds=[(-100, 100)] * dim)

        for _ in de:
            pass

        self.assertTrue(np.all(np.abs(de.best) < 1e-5))


class TestMultiObjective(unittest.TestCase):

    @parameterized.expand(all_strategies)
    def test_schaffer_n1(self, name):
        strategy, adaptivity = name.split("_")[1::2]
        strategy = EvolutionStrategy(strategy)
        de = NSDE(
            strategy=strategy, adaptivity=int(adaptivity)
        )
        de.init(schaffer_n1, bounds=[(-100, 100)])

        for _ in de:
            pass

        pareto = de.fit[de.fronts[0]]
        pareto = pareto[np.argsort(pareto[:, 0])]

        f1 = pareto[:, 0]
        f2 = (f1 ** 0.5 - 2) ** 2

        e = pareto[:, 1] - f2
        e_rel = e / np.abs(f2)
        rms = np.mean(e_rel ** 2) ** 0.5

        self.assertLess(rms, 1e-3)

    def test_binh_and_korn(self):
        de = NSDE()
        de.init(schaffer_n1, bounds=[(-100, 100)])

        for _ in de:
            pass

        pareto = de.fit[de.fronts[0]]
        pareto = pareto[np.argsort(pareto[:, 0])]

        f1 = pareto[:, 0]
        f2 = (f1 ** 0.5 - 2) ** 2

        e = pareto[:, 1] - f2
        e_rel = e / np.abs(f2)
        rms = np.mean(e_rel ** 2) ** 0.5

        self.assertLess(rms, 1e-3)


class TestLogic(unittest.TestCase):

    def test_seed_specified_repeatability(self):
        x = [None, None]

        for i in range(2):
            dim = 2
            de = NSDE(seed=11)
            de.init(paraboloid, bounds=[(-100, 100)] * dim)
            x[i] = np.copy(de.pop)
            del de

        self.assertTrue(np.all(x[0] == x[1]))

    def test_custom_population_size(self):
        dim = 2
        n_pop = 11
        de = NSDE(n_pop=n_pop)
        de.init(paraboloid, bounds=[(-100, 100)] * dim)
        self.assertEqual(de.pop.shape[0], n_pop)

    def test_zero_population_size(self):
        dim = 2
        de = NSDE(n_pop=0)
        de.init(paraboloid, bounds=[(-100, 100)] * dim)
        self.assertEqual(de.pop.shape[0], 5 * dim)
