import pytest

from pyvolutionary import OptimizationResult, FicksLawOptimization, FicksLawOptimizationConfig
from tests.fixtures import task


@pytest.fixture
def optimization_config() -> FicksLawOptimizationConfig:
    return FicksLawOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        C1=0.5,
        C2=2.0,
        C3=0.1,
        C4=0.2,
        C5=2.0,
        DD=0.01,
    )


def test_valid_optimization(optimization_config):
    o = FicksLawOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = FicksLawOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = FicksLawOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
