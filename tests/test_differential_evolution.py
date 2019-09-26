import numpy as np
import pytest
import itertools

from differential_evolution import *


def paraboloid(x):
    return np.sum(x * x)


@pytest.mark.parametrize(
    "strategy, adaptivity",
    list(
        itertools.product(
            [
                "/".join(s)
                for s in itertools.product(
                    *[
                        EvolutionStrategy.__mutation_strategies__.keys(),
                        ["1", "2", "3"],
                        EvolutionStrategy.__crossover_strategies__.keys(),
                        EvolutionStrategy.__repair_strategies__.keys(),
                    ]
                )
            ],
            [0, 1, 2],
        )
    ),
)
def test_differential_evolution(strategy, adaptivity):
    tol = 1e-8
    dim = 2

    strategy = EvolutionStrategy(strategy)
    de = DifferentialEvolution(strategy=strategy, tolx=tol, tolf=tol, adaptivity=adaptivity)
    de.init(paraboloid, bounds=[(-100, 100)] * dim)

    last_generation = None
    for last_generation in de:
        pass

    assert last_generation.dx < tol or last_generation.df < tol

    if strategy.__repr__().split("/")[0] != "best":
        # The "best" mutation strategy collapses prematurely sometimes, so we can't be sure this is always true
        assert np.all(last_generation.best < 1e-4 * np.ones_like(last_generation.best))
        assert last_generation.best_fit < 1e-4
