import pytest

from pyvolutionary import OptimizationResult, AntColonyOptimization, AntColonyOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> AntColonyOptimizationConfig:
    return AntColonyOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        archive_size=20,
        intent_factor=0.1,
        zeta=0.85,
    )


def test_valid_optimization(optimization_config):
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
