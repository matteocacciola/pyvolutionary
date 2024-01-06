import pytest

from pyvolutionary import OptimizationResult, InvasiveWeedOptimization, InvasiveWeedOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> InvasiveWeedOptimizationConfig:
    return InvasiveWeedOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        seed=[1, 4],
        exponent=2,
        sigma=[0.5, 0.1],
    )


def test_valid_optimization(optimization_config):
    o = InvasiveWeedOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = InvasiveWeedOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = InvasiveWeedOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
