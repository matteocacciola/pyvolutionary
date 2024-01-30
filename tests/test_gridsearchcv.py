import numpy as np
import pytest

from pyvolutionary import ContinuousVariable, Task, BiogeographyBasedOptimization, OptimizationResult, GridSearchCV


class Problem(Task):
    def objective_function(self, solution):
        return np.sum(np.array(solution) ** 2)


@pytest.fixture
def data() -> tuple[Task, dict[str, list]]:
    # Define the task with the bounds and the configuration of the optimizer
    task = Problem(variables=[ContinuousVariable(name=f"x{i}", lower_bound=-10, upper_bound=10) for i in range(0, 5)])

    paras_bbo_grid = {
        "max_cycles": [10, 20, 30, 40],
        "population_size": [50, 100, 150],
        "n_elites": [3, 4, 5, 6],
        "p_m": [0.01, 0.02, 0.05]
    }

    return task, paras_bbo_grid


def test_valid_serial(data):
    task, params_grid = data
    model = BiogeographyBasedOptimization()
    tuner = GridSearchCV(model, params_grid)

    tuner.execute(task=task)

    assert isinstance(tuner.best_row, dict)
    assert isinstance(tuner.best_score, float)
    assert isinstance(tuner.best_parameters, dict)

    g_best = tuner.resolve()

    assert isinstance(g_best, OptimizationResult)


def test_valid_parallel(data):
    task, params_grid = data
    model = BiogeographyBasedOptimization()
    tuner = GridSearchCV(model, params_grid)

    tuner.execute(task=task, n_trials=2, n_jobs=4)

    assert isinstance(tuner.best_row, dict)
    assert isinstance(tuner.best_score, float)
    assert isinstance(tuner.best_parameters, dict)

    g_best = tuner.resolve()

    assert isinstance(g_best, OptimizationResult)
