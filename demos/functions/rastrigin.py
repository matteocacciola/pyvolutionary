import numpy as np
from pyvolutionary import Task, ContinuousMultiVariable


class Rastrigin(Task):
    def objective_function(self, x: list[float]) -> float:
        A = 10
        return A + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


population = 100
generation = 400
fitness_error = 0.01
task = Rastrigin(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "rastrigin"
