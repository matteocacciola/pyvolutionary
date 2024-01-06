import pytest

from pyvolutionary import OptimizationResult, CoyotesOptimization, CoyotesOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> CoyotesOptimizationConfig:
    return CoyotesOptimizationConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        num_coyotes=3,
    )


def test_valid_optimization(optimization_config):
    o = CoyotesOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = CoyotesOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = CoyotesOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
