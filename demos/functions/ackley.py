import numpy as np
from pyvolutionary import Task, ContinuousVariable


class Ackley(Task):
    def objective_function(self, x: list[float]) -> float:
        A = 20
        B = 0.2
        C = 2 * np.pi
        return -A * np.exp(-B * np.sqrt(sum([xi ** 2 for xi in x]) / dimension)) - np.exp(
            sum([np.cos(C * xi) for xi in x]) / dimension
        ) + A + np.exp(1)


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 10e-4
task = Ackley(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "ackley"
