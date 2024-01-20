import pytest

from pyvolutionary import OptimizationResult, ElephantHerdOptimization, ElephantHerdOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ElephantHerdOptimizationConfig:
    return ElephantHerdOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        alpha=0.5,
        beta=0.5,
        n_clans=3,
    )


def test_valid_optimization(optimization_config):
    o = ElephantHerdOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ElephantHerdOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ElephantHerdOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
