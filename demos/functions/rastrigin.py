import numpy as np
from pyvolutionary import Task, ContinuousVariable


class Rastrigin(Task):
    def objective_function(self, x: list[float]) -> float:
        A = 10
        return A + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 0.01
task = Rastrigin(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "rastrigin"
