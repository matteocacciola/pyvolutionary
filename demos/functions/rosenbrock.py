from pyvolutionary import Task, ContinuousVariable


class Rosenbrock(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)])


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 0.01
task = Rosenbrock(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "rosenbrock"
