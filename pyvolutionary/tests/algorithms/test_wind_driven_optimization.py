import pytest

from pyvolutionary import OptimizationResult, WindDrivenOptimization, WindDrivenOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> WindDrivenOptimizationConfig:
    return WindDrivenOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        RT=2,
        g_c=0.2,
        alp=0.4,
        c_e=0.5,
        max_v=0.3,
    )


def test_valid_optimization(optimization_config):
    o = WindDrivenOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = WindDrivenOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = WindDrivenOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
