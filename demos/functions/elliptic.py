from pyvolutionary import Task, ContinuousMultiVariable


class Elliptic(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([(i + 1) * xi ** 2 for i, xi in enumerate(x)])


population = 100
generation = 400
fitness_error = 0.01
task = Elliptic(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-4.0, -4.0], upper_bounds=[4.0, 4.0])],
)
name = "elliptic"
