import pytest

from pyvolutionary import OptimizationResult, SpottedHyenaOptimization, SpottedHyenaOptimizationConfig
from tests.fixtures import task


@pytest.fixture
def optimization_config() -> SpottedHyenaOptimizationConfig:
    return SpottedHyenaOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        h_factor=5.0,
        n_trials=10,
    )


def test_valid_optimization(optimization_config):
    o = SpottedHyenaOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = SpottedHyenaOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = SpottedHyenaOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
