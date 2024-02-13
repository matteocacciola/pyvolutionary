import pytest

from pyvolutionary import OptimizationResult, HungerGamesSearchOptimization, HungerGamesSearchOptimizationConfig
from tests.fixtures import task


@pytest.fixture
def optimization_config() -> HungerGamesSearchOptimizationConfig:
    return HungerGamesSearchOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        PUP=0.08,
        LH=10000,
    )


def test_valid_optimization(optimization_config):
    o = HungerGamesSearchOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = HungerGamesSearchOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = HungerGamesSearchOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
