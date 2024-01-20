import pytest

from pyvolutionary import OptimizationResult, CamelCaravanOptimization, CamelCaravanOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> CamelCaravanOptimizationConfig:
    return CamelCaravanOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        burden_factor=0.5,
        death_rate=0.5,
        visibility=0.5,
        supply=10,
        endurance=10,
        temperatures=[-10, 10],
    )


def test_valid_optimization(optimization_config):
    o = CamelCaravanOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = CamelCaravanOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = CamelCaravanOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
