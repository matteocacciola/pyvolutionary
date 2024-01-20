import numpy as np
import pytest

from pyvolutionary import (
    Task,
    MultiObjectiveVariable,
    ForestOptimizationAlgorithm,
    ForestOptimizationAlgorithmConfig,
    OptimizationResult, ContinuousVariable,
)


class MultiObjectiveBenchmark(Task):
    # Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    def objective_function(self, solution):
        def booth(x, y):
            return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

        def bukin(x, y):
            return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)

        def matyas(x, y):
            return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        return [booth(solution[0], solution[1]), bukin(solution[0], solution[1]), matyas(solution[0], solution[1])]


@pytest.fixture
def data() -> tuple[ForestOptimizationAlgorithmConfig, Task]:
    # Define the task with the bounds and the configuration of the optimizer
    task = MultiObjectiveBenchmark(
        variables=MultiObjectiveVariable(name="x", lower_bounds=(-10, -10), upper_bounds=(10, 10)),
        objective_weights=[0.4, 0.1, 0.5],
    )

    configuration = ForestOptimizationAlgorithmConfig(
        population_size=200,
        fitness_error=10e-4,
        max_cycles=400,
        lifetime=5,
        area_limit=50,
        local_seeding_changes=1,
        global_seeding_changes=2,
        transfer_rate=0.5,
    )
    return configuration, task


def test_valid_optimization(data):
    optimization_config, task = data

    o = ForestOptimizationAlgorithm(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(data):
    optimization_config, task = data

    o = ForestOptimizationAlgorithm(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(data):
    optimization_config, task = data

    o = ForestOptimizationAlgorithm(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)


def test_fail_when_weights_are_less_than_objective_functions(data):
    optimization_config, task = data
    task.objective_weights = [0.1, 0.2]

    o = ForestOptimizationAlgorithm(optimization_config)
    with pytest.raises(ValueError):
        o.optimize(task)


def test_fail_when_weights_are_more_than_objective_functions(data):
    optimization_config, task = data
    task.objective_weights = [0.1, 0.2, 0.3, 0.4]

    o = ForestOptimizationAlgorithm(optimization_config)
    with pytest.raises(ValueError):
        o.optimize(task)


def test_fail_when_some_lower_bounds_are_not_lower_than_corresponding_upper_bounds():
    with pytest.raises(ValueError):
        MultiObjectiveBenchmark(
            variables=MultiObjectiveVariable(name="x", lower_bounds=(-10, 10), upper_bounds=(10, 10)),
            objective_weights=[0.4, 0.1, 0.5],
        )


def test_fail_when_multi_objective_weights_are_specified_but_variable_is_not_multi_objective():
    with pytest.raises(ValueError):
        MultiObjectiveBenchmark(
            variables=ContinuousVariable(name="x", lower_bound=-10, upper_bound=10),
            objective_weights=[0.4, 0.1, 0.5],
        )


def test_fail_when_number_of_lower_bounds_is_not_the_same_as_the_number_of_upper_bounds():
    with pytest.raises(ValueError):
        MultiObjectiveBenchmark(
            variables=MultiObjectiveVariable(name="x", lower_bounds=(-10), upper_bounds=(10, 10)),
            objective_weights=[0.4, 0.1, 0.5],
        )


def test_fail_when_some_weights_are_negative(data):
    with pytest.raises(ValueError):
        MultiObjectiveBenchmark(
            variables=MultiObjectiveVariable(name="x", lower_bounds=(-10, -10), upper_bounds=(10, 10)),
            objective_weights=[0.4, -0.1, 0.5],
        )

    optimization_config, task = data
    task.objective_weights = [0.4, -0.1, 0.2]

    o = ForestOptimizationAlgorithm(optimization_config)
    with pytest.raises(ValueError):
        o.optimize(task)
