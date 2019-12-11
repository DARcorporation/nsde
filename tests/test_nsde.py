#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nsde import *
from . import all_strategies, get_strategy_designation


def paraboloid(x):
    return np.sum(x * x)


def schaffer_n1(x):
    return [x ** 2, (x - 2) ** 2]


def binh_and_korn(x):
    return (
        [4 * np.sum(x ** 2), np.sum((x - 5) ** 2)],
        [((x[0] - 5) ** 2 + x[1] ** 2) / 25 - 1, 1 - ((x[0] - 8) ** 2 + (x[1] + 3) ** 2) / 7.7]
    )


def _get_de(mutation, number, crossover, repair, adaptivity):
    strategy = EvolutionStrategy(get_strategy_designation(mutation, number, crossover, repair))
    return NSDE(strategy=strategy, adaptivity=adaptivity, seed=11)


def _test_single_objective(fobj, x_opt, f_opt, *args):
    de = _get_de(*args)
    de.init(fobj, bounds=[(-100, 100)] * 2)

    for _ in de:
        pass

    try:
        assert de.best == pytest.approx(x_opt, rel=1e-1, abs=1e-2)
        assert de.best_fit == pytest.approx(f_opt, rel=1e-1, abs=1e-2)
    except AssertionError as e:
        if args[-1] == 0 or args[-2] == "clip":
            assert de.dx <= de.tolx or de.generation >= de.max_gen
        else:
            raise e


@all_strategies
def test_single_objective_unconstrained(*args):
    _test_single_objective(paraboloid, 0, 0, *args)


def test_single_objective_constrained(*args):
    def fobj(x):
        return paraboloid(x), 1 - x

    _test_single_objective(fobj, 1, 2, *args)


def test_single_objective_nan_landscape():
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

    assert de.best == pytest.approx(0.0, abs=1e-5)


@all_strategies
def test_multi_objective_unconstrained(*args):
    de = _get_de(*args)
    de.init(schaffer_n1, bounds=[(-100, 100)])

    for _ in de:
        pass

    pareto = de.fit[de.fronts[0]]
    pareto = pareto[np.argsort(pareto[:, 0])]

    f1 = pareto[:, 0]
    f2 = (f1 ** 0.5 - 2) ** 2

    assert pareto[:, 1] == pytest.approx(f2, rel=1e-2)


def test_multi_objective_constrained(*args):
    de = _get_de(*args)
    de.init(binh_and_korn, bounds=[(-15, 30)] * 2)

    for _ in de:
        pass
