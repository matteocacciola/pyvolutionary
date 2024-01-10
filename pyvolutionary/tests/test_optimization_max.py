import pytest

from pyvolutionary import (
    ContinuousVariable,
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
        max_cycles=100,
        p=[0.6, 0.4, 0.6],
        alpha=0.8,
        gamma=2.5,
    )
    dimension = 2
    task = Sphere(
        variables=[ContinuousVariable(name=f"x{i}", lower_bound=-10, upper_bound=10) for i in range(dimension)],
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
