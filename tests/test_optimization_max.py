import pytest

from pyvolutionary import (
    ContinuousMultiVariable,
    OptimizationResult,
    AfricanVultureOptimization,
    AfricanVultureOptimizationConfig,
    Task,
)


class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        return -sum(xi ** 2 for xi in x)


@pytest.fixture
def data() -> tuple[AfricanVultureOptimizationConfig, Task]:
    config = AfricanVultureOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        p=[0.6, 0.4, 0.6],
        alpha=0.8,
        gamma=2.5,
    )
    task = Sphere(
        variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10, -10], upper_bounds=[10, 10])],
        minmax="max",
    )

    return config, task


def test_valid_optimization(data):
    optimization_config, task = data

    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(data):
    optimization_config, task = data

    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(data):
    optimization_config, task = data

    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
