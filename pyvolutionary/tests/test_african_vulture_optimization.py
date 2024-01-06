import pytest

from pyvolutionary import OptimizationResult, AfricanVultureOptimization, AfricanVultureOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> AfricanVultureOptimizationConfig:
    return AfricanVultureOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        p=[0.6, 0.4, 0.6],
        alpha=0.8,
        gamma=2.5,
    )


def test_valid_optimization(optimization_config):
    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
