import numpy as np
from pyvolutionary import Task, ContinuousMultiVariable


class Griewank(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([xi ** 2 / 4000.0 for xi in x]) - np.prod(
            [np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)]
        ) + 1.0


population = 100
generation = 400
fitness_error = 0.01
task = Griewank(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "griewank"
