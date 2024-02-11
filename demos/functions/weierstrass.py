import numpy as np
from pyvolutionary import Task, ContinuousMultiVariable


class Weierstrass(Task):
    def objective_function(self, x: list[float]) -> float:
        a = 0.5
        b = 3
        return sum([a ** k * np.cos(b ** k * np.pi * xi) for k, xi in enumerate(x)])


population = 100
generation = 400
fitness_error = 0.01
task = Weierstrass(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "weierstrass"
