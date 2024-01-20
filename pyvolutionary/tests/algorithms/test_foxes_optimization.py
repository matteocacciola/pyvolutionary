import pytest

from pyvolutionary import OptimizationResult, FoxOptimization, FoxOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> FoxOptimizationConfig:
    return FoxOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        c1=0.18,
        c2=0.82,
    )


def test_valid_optimization(optimization_config):
    o = FoxOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = FoxOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = FoxOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
