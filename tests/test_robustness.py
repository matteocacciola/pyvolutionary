import glob
import importlib
import math
import re
import warnings

import numpy as np
import pytest

from pyvolutionary import ContinuousMultiVariable, DiscreteVariable, PermutationVariable, Task


class Rastrigin(Task):
    def objective_function(self, x):
        x = np.asarray(x, dtype=float)
        return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


class Zero(Task):
    def objective_function(self, x):
        return 0.0


class Mixed(Task):
    def objective_function(self, x):
        return float(np.sum(np.array(x[:3], dtype=float) ** 2) + [3.0, 0.5, 1.0][int(x[3])])


def algorithms() -> list:
    """
    Every algorithm, with the configuration of its own test module, shortened to a few cycles.
    """
    result = []
    for path in sorted(glob.glob("tests/algorithms/test_*.py")):
        module = importlib.import_module(path[:-3].replace("/", "."))
        algorithm = getattr(module, re.search(r"o = (\w+)\(", open(path).read()).group(1))
        config = module.optimization_config.__wrapped__().model_copy(update={"fitness_error": None, "max_cycles": 5})
        result.append(pytest.param(algorithm, config, id=algorithm.__name__))
    return result


def fingerprint(result) -> list:
    return [[(agent.cost, agent.position) for agent in population.agents] for population in result.evolution]


def is_finite(result) -> bool:
    return all(
        math.isfinite(agent.cost) and np.all(np.isfinite(np.array(agent.position, dtype=float)))
        for population in result.evolution for agent in population.agents
    )


@pytest.mark.parametrize("algorithm,config", algorithms())
def test_same_instance_is_reproducible(algorithm, config):
    # the state of a run must not leak into the next one
    task = Rastrigin(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10] * 4, upper_bounds=[10] * 4)], seed=3)
    optimizer = algorithm(config.model_copy(deep=True))
    first = fingerprint(optimizer.optimize(task))
    second = fingerprint(optimizer.optimize(task))
    assert first == second == fingerprint(algorithm(config.model_copy(deep=True)).optimize(task))


@pytest.mark.parametrize("algorithm,config", algorithms())
def test_all_costs_equal_to_zero(algorithm, config):
    # e.g. a plateau, or the optimum reached by the whole population
    task = Zero(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-1] * 4, upper_bounds=[1] * 4)], seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = algorithm(config.model_copy(deep=True)).optimize(task)
    assert result.best_solution.cost == 0


@pytest.mark.parametrize("algorithm,config", algorithms())
def test_mixed_variables_stay_finite(algorithm, config):
    task = Mixed(variables=[
        ContinuousMultiVariable(name="x", lower_bounds=[-5] * 3, upper_bounds=[5] * 3),
        DiscreteVariable(name="d", choices=["a", "b", "c"]),
    ], seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = algorithm(config.model_copy(deep=True)).optimize(task)
    assert is_finite(result)


class Tsp(Task):
    def objective_function(self, x):
        route = self.transform_solution(x)["route"]
        cities = self.data["cities"]
        return float(sum(np.hypot(*np.subtract(cities[a], cities[b])) for a, b in zip(route, route[1:] + route[:1])))


@pytest.mark.parametrize("algorithm,config", algorithms())
def test_permutation_task(algorithm, config):
    cities = np.random.RandomState(7).uniform(0, 100, (8, 2)).tolist()
    task = Tsp(variables=[PermutationVariable(name="route", items=list(range(8)))], data={"cities": cities}, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = algorithm(config.model_copy(deep=True)).optimize(task)
    assert sorted(task.transform_solution(result.best_solution.position)["route"]) == list(range(8))
    assert is_finite(result)


@pytest.mark.parametrize("algorithm,config", algorithms())
def test_reported_costs_are_consistent(algorithm, config):
    # every agent reports the cost of its own position, and the best solution is the best agent found
    task = Rastrigin(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-5] * 4, upper_bounds=[5] * 4)], seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = algorithm(config.model_copy(deep=True)).optimize(task)
    agents = [agent for population in result.evolution for agent in population.agents]
    for agent in agents:
        assert agent.cost == pytest.approx(task.solve(agent.position), rel=1e-9, abs=1e-9)
    assert result.best_solution.cost == pytest.approx(min(agent.cost for agent in agents), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("algorithm,config", algorithms())
def test_small_populations_and_dimensions(algorithm, config):
    # any population from the minimum size of the algorithm on, and a single dimension, work; below the minimum size,
    # the algorithm refuses the population with a clear message
    minimum = algorithm(config.model_copy(deep=True))._minimum_population_size()
    for population_size in (minimum, minimum + 1, 7):
        for dims in (1, 2):
            small = config.model_copy(update={"population_size": population_size})
            task = Rastrigin(
                variables=[ContinuousMultiVariable(name="x", lower_bounds=[-5] * dims, upper_bounds=[5] * dims)], seed=0
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = algorithm(small).optimize(task)
            assert is_finite(result)
    if minimum > 2:
        with pytest.raises(ValueError, match="needs a population of at least"):
            algorithm(config.model_copy(update={"population_size": minimum - 1})).optimize(
                Rastrigin(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-5], upper_bounds=[5])])
            )
