from pyvolutionary import Task, ContinuousMultiVariable


class Zakharov(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([xi ** 2 for xi in x]) + (
            0.5 * sum([i * xi for i, xi in enumerate(x)])
        ) ** 2 + (0.5 * sum([i * xi for i, xi in enumerate(x)])) ** 4


population = 100
generation = 400
fitness_error = 0.01
task = Zakharov(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "zakharov"
