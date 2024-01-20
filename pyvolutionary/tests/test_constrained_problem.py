import numpy as np
import pytest

from pyvolutionary import Task, ContinuousVariable, AntLionOptimization, AntLionOptimizationConfig, OptimizationResult


class ConstrainedBenchmark(Task):
    def objective_function(self, solution):
        def g1(x):
            return 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10

        def g2(x):
            return 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10

        def g3(x):
            return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10

        def g4(x):
            return -8 * x[0] + x[9]

        def g5(x):
            return -8 * x[1] + x[10]

        def g6(x):
            return -8 * x[2] + x[11]

        def g7(x):
            return -2 * x[3] - x[4] + x[9]

        def g8(x):
            return -2 * x[5] - x[6] + x[10]

        def g9(x):
            return -2 * x[7] - x[8] + x[11]

        def violate(value):
            return 0 if value <= 0 else value

        solution = np.array(solution)
        fx = 5 * np.sum(solution[:4]) - 5 * np.sum(solution[:4] ** 2) - np.sum(solution[4:])

        fx += violate(g1(solution)) ** 2 + violate(g2(solution)) + violate(g3(solution)) + (
            2 * violate(g4(solution)) + violate(g5(solution)) + violate(g6(solution))
        ) + violate(g7(solution)) + violate(g8(solution)) + violate(g9(solution))
        return fx


@pytest.fixture
def data() -> tuple[AntLionOptimizationConfig, Task]:
    # Define the task with the bounds and the configuration of the optimizer
    task = ConstrainedBenchmark(
        variables=(
            [ContinuousVariable(name=f"x{i}", lower_bound=0, upper_bound=1) for i in range(9)] + [
                ContinuousVariable(name=f"x{i}", lower_bound=0, upper_bound=100) for i in range(9, 12)
            ] + [ContinuousVariable(name=f"x12", lower_bound=0, upper_bound=1)]
        )
    )

    configuration = AntLionOptimizationConfig(population_size=200, fitness_error=10e-4, max_cycles=400)
    return configuration, task


def test_valid_optimization(data):
    optimization_config, task = data

    o = AntLionOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(data):
    optimization_config, task = data

    o = AntLionOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(data):
    optimization_config, task = data

    o = AntLionOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
