import numpy as np

from differential_evolution import *


def paraboloid(x):
    return np.sum(x * x)


def test_differential_evolution():
    dim = 2

    strategy = EvolutionStrategy("rand/1/exp")
    de = DifferentialEvolution(strategy=strategy, max_gen=1000, tolx=1e-8, tolf=1e-8)
    de.init(paraboloid, bounds=[(-100, 100)] * dim)

    last_generation = None
    for last_generation in de:
        pass

    assert np.all(last_generation.best < 1e-4 * np.ones_like(last_generation.best))
    assert last_generation.best_fit < 1e-4
