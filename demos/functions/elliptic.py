from pyvolutionary import Task, ContinuousVariable


class Elliptic(Task):
    def objective_function(self, x: list[float]) -> float:
        return sum([(i + 1) * xi ** 2 for i, xi in enumerate(x)])


population = 100
dimension = 2
position_min = -4.0
position_max = 4.0
generation = 400
fitness_error = 0.01
task = Elliptic(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)
name = "elliptic"
