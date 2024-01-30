from typing import Any
import numpy as np
import pytest

from pyvolutionary import Task, PermutationVariable, VirusColonySearchOptimization
from pyvolutionary.helpers import distance


class TspProblem(Task):
    def objective_function(self, x: list[Any]) -> float:
        x_transformed = self.transform_solution(x)
        routes = x_transformed["routes"]
        city_pos = self.data["city_positions"]
        n_routes = len(routes)
        return np.sum([distance(
            city_pos[route], city_pos[routes[(idx + 1) % n_routes]]
        ) for idx, route in enumerate(routes)])


@pytest.fixture
def data() -> Task:
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

    return task


def test_not_passed_configuration(data):
    task = data
    o = VirusColonySearchOptimization()

    with pytest.raises(ValueError):
        o.optimize(task)
