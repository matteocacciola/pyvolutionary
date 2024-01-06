import numpy as np
from pyvolutionary import Task, ContinuousVariable


class Griewank(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([xi ** 2 / 4000.0 for xi in x]) - np.prod(
            [np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)]
        ) + 1.0


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 10e-4
task = Griewank(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "griewank"
