import math
import numpy as np

from pyvolutionary import (
    ContinuousMultiVariable,
    DiscreteVariable,
    EarlyStopping,
    GreyWolfOptimization,
    GreyWolfOptimizationConfig,
    Multitask,
    Task,
    TaskType,
    WaterCycleOptimization,
    WaterCycleOptimizationConfig,
)
from pyvolutionary.enums import ModeSolver
from pyvolutionary.helpers import best_agent_formatted, distances, get_levy_flight_step, random_selection
from pyvolutionary.models import Agent
from tests.fixtures import Rastrigin


class Sphere(Task):
    def objective_function(self, x):
        return float(np.sum(np.asarray(x) ** 2))


class TwoObjectives(Task):
    def objective_function(self, x):
        return [float(np.sum(np.asarray(x) ** 2)), float(np.sum(np.abs(x)))]


def make_task(**kwargs) -> Task:
    return Sphere(
        variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10] * 5, upper_bounds=[10] * 5)], **kwargs
    )


def make_config(**kwargs) -> GreyWolfOptimizationConfig:
    return GreyWolfOptimizationConfig(population_size=10, fitness_error=None, max_cycles=5, **kwargs)


def test_optimize_twice_on_same_instance():
    task = make_task(seed=42)
    optimizer = GreyWolfOptimization(make_config())
    first = optimizer.optimize(task)
    second = optimizer.optimize(task)

    assert len(second.evolution) == len(first.evolution)
    assert second.rates == first.rates
    assert second.best_solution.cost == first.best_solution.cost


def test_process_mode_generates_distinct_agents():
    task = make_task()
    optimizer = GreyWolfOptimization(make_config())
    optimizer._task = task
    optimizer._mode = ModeSolver.PROCESS
    optimizer._workers = 4

    population = optimizer._generate_agents(20)
    assert len({tuple(agent.position) for agent in population}) == 20


def test_multi_objective_maximization():
    task = TwoObjectives(
        variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10] * 3, upper_bounds=[10] * 3)],
        objective_weights=[0.5, 0.5],
        minmax=TaskType.MAX,
    )
    result = GreyWolfOptimization(make_config()).optimize(task)
    assert result.best_solution.cost > 0


def test_correct_solution_clips_to_bounds():
    task = make_task()
    assert task.correct_solution([-20, -10, 0, 10, 20]) == [-10.0, -10.0, 0.0, 10.0, 10.0]

    mixed = Sphere(variables=[
        ContinuousMultiVariable(name="x", lower_bounds=[-1, -1], upper_bounds=[1, 1]),
        DiscreteVariable(name="d", choices=["a", "b", "c"]),
    ])
    assert mixed.correct_solution([-3, 0.5, 7.9]) == [-1.0, 0.5, 2]


def test_amend_solution():
    task = make_task()
    solution = np.array([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(task.amend_solution(solution), solution)

    amended = task.amend_solution([0, 1, 2, 3, 40])
    np.testing.assert_array_equal(amended[:4], solution[:4])
    assert -10 <= amended[4] <= 10


def test_increase_solution_moves_from_current_position():
    task = make_task()
    solution = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 100
    increased = task.increase_solution(solution, scale_factor)
    # the increment is bounded by the bounds of the search space, divided by the scale factor
    assert np.all(np.abs(increased - np.array(solution)) <= 10 / scale_factor)


def test_random_selection_with_rounding_errors():
    p = np.full(3, 1 / 3 - 1e-12)
    for _ in range(100):
        assert 0 <= random_selection(p) <= 2


def test_best_agent_formatted_does_not_alter_population():
    population = [
        Agent(position=[1.0, 2.0, 3.0, 4.0], cost=1.0, fitness=0.5),
        Agent(position=[5.0, 6.0, 7.0, 8.0], cost=2.0, fitness=0.3),
    ]
    best = best_agent_formatted(population, 2)
    assert np.shape(best.position) == (2, 2)
    assert population[0].position == [1.0, 2.0, 3.0, 4.0]


def test_early_stopping_on_stagnation():
    task = make_task(seed=1)
    config = GreyWolfOptimizationConfig(
        population_size=10, fitness_error=None, max_cycles=1000, early_stopping=EarlyStopping(patience=3, min_delta=1.0)
    )
    result = GreyWolfOptimization(config).optimize(task)
    # an improvement of the error by at least 1.0 is impossible: the optimization stops after (1 + patience) cycles
    assert len(result.evolution) == 1 + 4


def test_multitask_with_one_mode_per_algorithm():
    multitask = Multitask(
        algorithms=(GreyWolfOptimization(make_config()), GreyWolfOptimization(make_config())),
        tasks=(make_task(), make_task(), make_task()),
        modes=("serial", "thread"),
    )
    multitask.execute(n_trials=1, n_jobs=1)
    assert len(multitask._df2) == 2


def test_water_cycle_assigns_streams_to_every_river():
    # with this seed, the share of streams of a river was rounded to zero, and the empty stream made the step crash
    task = Rastrigin(
        variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10] * 3, upper_bounds=[10] * 3)], seed=137
    )
    config = WaterCycleOptimizationConfig(population_size=20, fitness_error=0.01, max_cycles=10, nsr=4, wc=2.0)
    WaterCycleOptimization(config).optimize(task)


def test_distances():
    points = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]
    np.testing.assert_allclose(distances(points), [[0, 5, 10], [5, 0, 5], [10, 5, 0]])


def test_levy_flight_step_scale():
    np.random.seed(0)
    beta = 1.5
    sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
        math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    )) ** (1 / beta)
    # with multiplier 1 and case -1, the step is u / |v|^(1/beta): u ~ N(0, sigma_u) and v ~ N(0, 1)
    steps = get_levy_flight_step(beta=beta, multiplier=1.0, size=200000, case=-1)
    np.random.seed(0)
    u = np.random.normal(0, sigma_u, 200000)
    v = np.random.normal(0, 1, 200000)
    np.testing.assert_allclose(steps, u / np.abs(v) ** (1 / beta))

    assert np.isscalar(get_levy_flight_step(beta=beta, case=-1))
    assert np.shape(get_levy_flight_step(beta=beta, size=np.array(3), case=-1)) == (3,)


def test_water_cycle_with_zero_cost():
    class Zero(Task):
        def objective_function(self, x):
            return 0.0

    task = Zero(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-1] * 2, upper_bounds=[1] * 2)], seed=1)
    config = WaterCycleOptimizationConfig(population_size=20, fitness_error=None, max_cycles=3, nsr=4, wc=2.0)
    result = WaterCycleOptimization(config).optimize(task)
    assert result.best_solution.cost == 0.0
