import pytest

from pyvolutionary import OptimizationResult, CoralReefOptimization, CoralReefOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> CoralReefOptimizationConfig:
    return CoralReefOptimizationConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        po=0.3,
        Fb=0.8,
        Fa=0.1,
        Fd=0.1,
        Pd=0.3,
        GCR=0.1,
        gamma=[0.01, 0.5],
        n_trials=5,
    )


def test_valid_optimization(optimization_config):
    o = CoralReefOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = CoralReefOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = CoralReefOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
