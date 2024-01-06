import numpy as np
from pyvolutionary import Task, ContinuousVariable


class Michalewicz(Task):
    def objective_function(self, x: list[float]) -> float:
        m = 10
        return -sum([np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi)**(2 * m) for i, xi in enumerate(x)])


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 10e-4
task = Michalewicz(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "michalewicz"
