import pytest

from pyvolutionary import OptimizationResult, ArchimedeOptimization, ArchimedeOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ArchimedeOptimizationConfig:
    return ArchimedeOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        c1=2.0,
        c2=2.0,
        c3=2.0,
        c4=0.5,
        acc=[0.2, 0.9],
    )


def test_valid_optimization(optimization_config):
    o = ArchimedeOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ArchimedeOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ArchimedeOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
