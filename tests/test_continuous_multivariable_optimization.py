import pytest

from pyvolutionary import (
    OptimizationResult,
    AfricanVultureOptimization,
    AfricanVultureOptimizationConfig,
    ContinuousMultiVariable,
    Task,
)
from tests.fixtures import Rastrigin


@pytest.fixture
def data() -> tuple[Task, AfricanVultureOptimizationConfig]:
    task = Rastrigin(
        variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10, -10, -10], upper_bounds=[10, 10, 10])],
    )
    config = AfricanVultureOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        p=[0.6, 0.4, 0.6],
        alpha=0.8,
        gamma=2.5,
    )

    return task, config


def test_valid_optimization(data):
    task, optimization_config = data
    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(data):
    task, optimization_config = data
    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(data):
    task, optimization_config = data
    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
