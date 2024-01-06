import pytest

from pyvolutionary import OptimizationResult, FireworksOptimization, FireworksOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> FireworksOptimizationConfig:
    return FireworksOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        sparks_num=50,
        a=0.04,
        b=0.8,
        explosion_amplitude=40,
        gaussian_explosion_number=5,
    )


def test_valid_optimization(optimization_config):
    o = FireworksOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = FireworksOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = FireworksOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
