import pytest

from pyvolutionary import OptimizationResult, GrasshopperOptimization, GrasshopperOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> GrasshopperOptimizationConfig:
    return GrasshopperOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        c_min=0.00004,
        c_max=2.0,
    )


def test_valid_optimization(optimization_config):
    o = GrasshopperOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = GrasshopperOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = GrasshopperOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
