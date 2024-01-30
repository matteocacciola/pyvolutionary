import numpy as np
from pyvolutionary.models import Task, ContinuousVariable


class Rastrigin(Task):
    def objective_function(self, x: list[float]) -> float:
        A = 10
        n = len(x)
        sum_term = sum(xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x)
        return A * n + sum_term


class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum(xi ** 2 for xi in x)


dimension = 3
task = Rastrigin(
    variables=[ContinuousVariable(name=f"x{i}", lower_bound=-10, upper_bound=10) for i in range(dimension)],
)
