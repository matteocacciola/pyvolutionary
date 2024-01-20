import pytest

from pyvolutionary import OptimizationResult, BatOptimization, BatOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> BatOptimizationConfig:
    return BatOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        loudness=[1, 2],
        pulse_rate=[0.1, 0.9],
        pulse_frequency=[-0.5, 0.5],
    )


def test_valid_optimization(optimization_config):
    o = BatOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = BatOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = BatOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
