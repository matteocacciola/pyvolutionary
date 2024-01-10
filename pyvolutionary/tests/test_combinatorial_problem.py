from typing import Any
import numpy as np
import pytest

from pyvolutionary import (
    Task,
    PermutationVariable,
    VirusColonySearchOptimization,
    VirusColonySearchOptimizationConfig,
    OptimizationResult,
)
from pyvolutionary.helpers import distance


class TspProblem(Task):
    def objective_function(self, x: list[Any]) -> float:
        x_transformed = self.transform_position(x)
        routes = x_transformed["routes"]
        city_pos = self.data["city_positions"]
        n_routes = len(routes)
        return np.sum([distance(
            city_pos[route], city_pos[routes[(idx + 1) % n_routes]]
        ) for idx, route in enumerate(routes)])


@pytest.fixture
def data() -> tuple[VirusColonySearchOptimizationConfig, Task]:
    city_positions = [
        [60, 200], [180, 200], [80, 180], [140, 180], [20, 160],
        [100, 160], [200, 160], [140, 140], [40, 120], [100, 120],
        [180, 100], [60, 80], [120, 80], [180, 60], [20, 40],
        [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]
    ]
    task = TspProblem(
        variables=[PermutationVariable(name="routes", items=list(range(0, len(city_positions))))],
        data={"city_positions": city_positions},
    )

    config = VirusColonySearchOptimizationConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        lamda=0.1,
        sigma=2.5,
    )

    return config, task


def test_valid_optimization(data):
    optimization_config, task = data

    o = VirusColonySearchOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(data):
    optimization_config, task = data

    o = VirusColonySearchOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(data):
    optimization_config, task = data

    o = VirusColonySearchOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
