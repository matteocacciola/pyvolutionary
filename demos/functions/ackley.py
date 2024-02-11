import numpy as np
from pyvolutionary import Task, ContinuousMultiVariable


class Ackley(Task):
    def objective_function(self, x: list[float]) -> float:
        A = 20
        B = 0.2
        C = 2 * np.pi
        dimension = len(x)
        return -A * np.exp(-B * np.sqrt(sum([xi ** 2 for xi in x]) / dimension)) - np.exp(
            sum([np.cos(C * xi) for xi in x]) / dimension
        ) + A + np.exp(1)


population = 100
generation = 400
fitness_error = 0.01
task = Ackley(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "ackley"
