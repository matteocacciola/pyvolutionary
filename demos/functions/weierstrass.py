import numpy as np
from pyvolutionary import Task, ContinuousVariable


class Weierstrass(Task):
    def objective_function(self, x: list[float]) -> float:
        a = 0.5
        b = 3
        return sum([a ** k * np.cos(b ** k * np.pi * xi) for k, xi in enumerate(x)])


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 0.01
task = Weierstrass(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "weierstrass"
