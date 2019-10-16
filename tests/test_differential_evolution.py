#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from differential_evolution import *


def paraboloid(x):
    return np.sum(x * x)


@pytest.mark.parametrize("repair", EvolutionStrategy.__repair_strategies__.keys())
@pytest.mark.parametrize("crossover", EvolutionStrategy.__crossover_strategies__.keys())
@pytest.mark.parametrize("number", [1, 2, 3])
@pytest.mark.parametrize("mutation", EvolutionStrategy.__mutation_strategies__.keys())
@pytest.mark.parametrize("adaptivity", [0, 1, 2])
def test_differential_evolution(mutation, number, crossover, repair, adaptivity):
    tol = 1e-8
    dim = 2

    strategy = EvolutionStrategy("/".join([mutation, str(number), crossover, repair]))
    de = DifferentialEvolution(
        strategy=strategy, tolx=tol, tolf=tol, adaptivity=adaptivity
    )
    de.init(paraboloid, bounds=[(-100, 100)] * dim)

    last_generation = None
    for last_generation in de:
        pass

    try:
        assert np.all(last_generation.best < 1e-4 * np.ones_like(last_generation.best))
        assert last_generation.best_fit < 1e-4
    except AssertionError:
        # This is to account for strategies sometimes 'collapsing' prematurely.
        # This is not a failed test, this is a known phenomenon with DE.
        # In this case we just check that one of the two tolerances was triggered.
        assert last_generation.dx < tol or last_generation.df < tol


def test_seed_specified_repeatability():
    x = [None, None]

    for i in range(2):
        dim = 2
        de = DifferentialEvolution(seed=11)
        de.init(paraboloid, bounds=[(-100, 100)] * dim)
        x[i] = np.copy(de.pop)
        del de

    assert np.all(x[0] == x[1])


def test_custom_population_size():
    dim = 2
    n_pop = 11
    de = DifferentialEvolution(n_pop=n_pop)
    de.init(paraboloid, bounds=[(-100, 100)]*dim)

    assert de.pop.shape[0] == n_pop
