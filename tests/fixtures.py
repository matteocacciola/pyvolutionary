import numpy as np
from pyvolutionary.models import Task, ContinuousMultiVariable


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
lower_bounds = np.repeat(-10, dimension).tolist()
upper_bounds = np.repeat(10, dimension).tolist()
task = Rastrigin(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=lower_bounds, upper_bounds=upper_bounds)],
)
