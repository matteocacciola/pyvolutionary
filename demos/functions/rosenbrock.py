from pyvolutionary import Task, ContinuousMultiVariable


class Rosenbrock(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)])


population = 100
generation = 400
fitness_error = 0.01
task = Rosenbrock(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "rosenbrock"
