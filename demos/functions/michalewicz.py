import numpy as np
from pyvolutionary import Task, ContinuousMultiVariable


class Michalewicz(Task):
    def objective_function(self, x: list[float]) -> float:
        m = 10
        return -sum([np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi)**(2 * m) for i, xi in enumerate(x)])


population = 100
generation = 400
fitness_error = 0.01
task = Michalewicz(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "michalewicz"
