import pytest

from pyvolutionary import OptimizationResult, FlowerPollinationAlgorithmOptimization, FlowerPollinationAlgorithmOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> FlowerPollinationAlgorithmOptimizationConfig:
    return FlowerPollinationAlgorithmOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        p_s=0.8,
        levy_multiplier=0.2,
    )


def test_valid_optimization(optimization_config):
    o = FlowerPollinationAlgorithmOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = FlowerPollinationAlgorithmOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = FlowerPollinationAlgorithmOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
