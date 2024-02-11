from pyvolutionary import Task, ContinuousMultiVariable


class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        x1, x2 = x
        f1 = x1 - 2 * x2 + 3
        f2 = 2 * x1 + x2 - 8
        return f1 ** 2 + f2 ** 2


population = 200
generation = 400
fitness_error = 0.01
task = Sphere(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=[-100.0, -100.0], upper_bounds=[100.0, 100.0])],
)
name = "sphere"
