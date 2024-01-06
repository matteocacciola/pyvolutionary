from pyvolutionary import Task, ContinuousVariable


class Zakharov(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([xi ** 2 for xi in x]) + (
            0.5 * sum([i * xi for i, xi in enumerate(x)])
        ) ** 2 + (0.5 * sum([i * xi for i, xi in enumerate(x)])) ** 4


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 10e-4
task = Zakharov(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "zakharov"
