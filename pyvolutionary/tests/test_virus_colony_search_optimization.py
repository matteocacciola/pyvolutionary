import pytest

from pyvolutionary import OptimizationResult, VirusColonySearchOptimization, VirusColonySearchOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> VirusColonySearchOptimizationConfig:
    return VirusColonySearchOptimizationConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        lamda=0.1,
        sigma=2.5,
    )


def test_valid_optimization(optimization_config):
    o = VirusColonySearchOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = VirusColonySearchOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = VirusColonySearchOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
