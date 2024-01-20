import pytest

from pyvolutionary import OptimizationResult, ForestOptimizationAlgorithm, ForestOptimizationAlgorithmConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ForestOptimizationAlgorithmConfig:
    return ForestOptimizationAlgorithmConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        lifetime=10,
        area_limit=5,
        local_seeding_changes=1,
        global_seeding_changes=3,
        transfer_rate=0.5,
    )


def test_valid_optimization(optimization_config):
    o = ForestOptimizationAlgorithm(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ForestOptimizationAlgorithm(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ForestOptimizationAlgorithm(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
