#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from nsde import EvolutionStrategy


def all_strategies(fun):
    @pytest.mark.parametrize("adaptivity", [0, 1, 2])
    @pytest.mark.parametrize("repair", EvolutionStrategy.__repair_strategies__.keys())
    @pytest.mark.parametrize("crossover", EvolutionStrategy.__crossover_strategies__.keys())
    @pytest.mark.parametrize("number", [1, 2])
    @pytest.mark.parametrize("mutation", EvolutionStrategy.__mutation_strategies__.keys())
    def f(mutation, number, crossover, repair, adaptivity):
        return fun(mutation, number, crossover, repair, adaptivity)
    return f


def get_strategy_designation(mutation, number, crossover, repair):
    return "/".join([mutation, str(number), crossover, repair])
