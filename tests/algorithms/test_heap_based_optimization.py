import pytest

from pyvolutionary import OptimizationResult, HeapBasedOptimization, HeapBasedOptimizationConfig
from tests.fixtures import task


@pytest.fixture
def optimization_config() -> HeapBasedOptimizationConfig:
    return HeapBasedOptimizationConfig(population_size=20, fitness_error=0.01, max_cycles=10, degree=2)


def test_valid_optimization(optimization_config):
    o = HeapBasedOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = HeapBasedOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = HeapBasedOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
