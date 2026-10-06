import math
import numpy as np
import pytest

from pyvolutionary import (
    BeeColonyOptimization,
    BeeColonyOptimizationConfig,
    CoralReefOptimization,
    EarthwormsOptimization,
    EarthwormsOptimizationConfig,
    FireflySwarmOptimization,
    FireflySwarmOptimizationConfig,
    FishSchoolSearchOptimization,
    FishSchoolSearchOptimizationConfig,
    ContinuousMultiVariable,
    DiscreteMultiVariable,
    DiscreteVariable,
    EarlyStopping,
    GreyWolfOptimization,
    GreyWolfOptimizationConfig,
    ImperialistCompetitiveOptimization,
    ImperialistCompetitiveOptimizationConfig,
    Multitask,
    PermutationVariable,
    Task,
    TaskType,
    WaterCycleOptimization,
    WaterCycleOptimizationConfig,
)
from pyvolutionary.enums import ModeSolver
from pyvolutionary.helpers import (
    best_agent_formatted, distances, get_levy_flight_step, normalize_costs, random_selection
)
from pyvolutionary.models import Agent, Variable
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


def test_bee_colony_and_firefly_do_not_alter_the_configuration():
    task = make_task(seed=3)
    for optimizer_class, config in [
        (BeeColonyOptimization, BeeColonyOptimizationConfig(
            population_size=20, fitness_error=None, max_cycles=5, scouting_limit=5
        )),
        (FireflySwarmOptimization, FireflySwarmOptimizationConfig(
            population_size=10, fitness_error=None, max_cycles=5, alpha=0.5, beta_min=0.2, gamma=0.99
        )),
    ]:
        original = config.model_copy()
        first = optimizer_class(config).optimize(task)
        second = optimizer_class(config).optimize(task)

        assert config == original
        assert second.best_solution.cost == first.best_solution.cost


def make_imperialist(**kwargs) -> ImperialistCompetitiveOptimization:
    config = ImperialistCompetitiveOptimizationConfig(**{
        "population_size": 5,
        "fitness_error": None,
        "max_cycles": 5,
        "assimilation_rate": 0.4,
        "revolution_rate": 0.5,
        "alpha_rate": 0.8,
        "revolution_probability": 0.9,
        "number_of_countries": 20,
        **kwargs,
    })
    return ImperialistCompetitiveOptimization(config)


def all_countries(optimizer: ImperialistCompetitiveOptimization) -> list:
    empires = optimizer._ImperialistCompetitiveOptimization__empires
    return [country for empire in empires for country in [empire.emperor] + empire.colonies]


def test_imperialist_initial_countries_are_unique():
    # 3 x 3 discrete combinations: the 9 countries must be all the different combinations
    task = Sphere(
        variables=[DiscreteVariable(name="a", choices=[0, 1, 2]), DiscreteVariable(name="b", choices=[0, 1, 2])], seed=0
    )
    optimizer = make_imperialist(population_size=3, number_of_countries=9, max_cycles=1)
    optimizer._task = task
    optimizer._init_population()
    representations = {tuple(c.representation) for c in optimizer._ImperialistCompetitiveOptimization__countries}
    assert len(representations) == 9

    # more countries than distinct combinations: the initialization still ends
    optimizer = make_imperialist(population_size=3, number_of_countries=12, max_cycles=1)
    optimizer._task = task
    optimizer._init_population()
    assert len(optimizer._ImperialistCompetitiveOptimization__countries) == 12


def test_imperialist_countries_keep_cost_consistent_with_representation():
    # a rejected revolution must not alter the colony: each cost must match its representation
    task = make_task(seed=5)
    optimizer = make_imperialist()
    optimizer.optimize(task)
    for country in all_countries(optimizer):
        assert country.cost == task.solve(country.representation)


def test_imperialist_with_one_dimension():
    task = Sphere(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10], upper_bounds=[10])], seed=0)
    result = make_imperialist().optimize(task)
    assert len(result.evolution) == 6


def test_imperialist_power_weights():
    optimizer = make_imperialist()
    # the strongest empire (lowest cost) has the highest weight, also with negative costs (e.g. of maximization tasks)
    for costs in ([10.0, 5.0, 1.0], [-10.0, -5.0, -1.0], [-3.0, 0.5, 4.0]):
        weights = optimizer._power_weights(np.array(costs))
        assert np.all(np.isfinite(weights))
        assert np.argmax(weights) == np.argmin(costs)


def test_coral_reef_without_configuration():
    # as the other algorithms, it can be created without a configuration (e.g. for the HyperTuner)
    CoralReefOptimization()


def test_earthworms_with_one_dimension():
    task = Sphere(variables=[ContinuousMultiVariable(name="x", lower_bounds=[-10], upper_bounds=[10])], seed=0)
    config = EarthwormsOptimizationConfig(
        population_size=10, fitness_error=None, max_cycles=5, alpha=0.98, beta=0.9, gamma=0.9, keep=2, prob_mutate=0.05,
        prob_crossover=0.8,
    )
    assert len(EarthwormsOptimization(config).optimize(task).evolution) == 6


def test_normalize_costs_keeps_the_order():
    for costs in ([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [-2.0, 1.0, 1.0], [0.0, 0.0, 0.0]):
        normalized = normalize_costs(np.array(costs))
        assert np.all(np.isfinite(normalized))
        assert np.array_equal(np.argsort(normalized, kind="stable"), np.argsort(costs, kind="stable"))


def test_discrete_multi_variable_bounds():
    for n in (1, 2, 3):
        task = Sphere(variables=[DiscreteMultiVariable(name="d", choices=[["a", "b", "c"], ["x", "y"], [1, 2, 3, 4]][:n])])
        lb, ub = task.get_bounds()
        np.testing.assert_array_equal(lb, [0] * n)
        np.testing.assert_array_equal(ub, [2, 1, 3][:n])


def test_transform_solution_of_one_dimensional_multi_variable():
    task = Sphere(variables=[ContinuousMultiVariable(name="x", lower_bounds=[0], upper_bounds=[1])])
    assert task.transform_solution([0.5]) == {"x": [0.5]}


def test_permutation_variable():
    task = Sphere(variables=[PermutationVariable(name="route", items=["a", "b", "c", "d"])])
    # one dimension per item: the route is the order of the items by increasing key
    assert task.space_dimension == 4
    keys = [0.3, 2.9, 0.1, 1.5]
    corrected = task.correct_solution(keys)
    assert corrected == [1, 3, 0, 2]
    assert task.correct_solution(corrected) == corrected
    assert task.transform_solution(keys) == task.transform_solution(corrected) == {"route": ["c", "a", "d", "b"]}


def test_task_follows_its_variables():
    task = make_task()
    other = task.model_copy(update={
        "variables": [ContinuousMultiVariable(name="x", lower_bounds=[0] * 2, upper_bounds=[1] * 2)]
    })
    assert other.space_dimension == 2
    np.testing.assert_array_equal(other.get_bounds()[1], [1, 1])
    assert other.correct_solution([5, -5]) == [1.0, 0.0]
    # the original task is left as it is
    assert task.space_dimension == 5
    np.testing.assert_array_equal(task.get_bounds()[1], [10] * 5)


class Rounded(Variable):
    """
    A custom variable whose correction is not idempotent: it must be applied as before, i.e. also when evaluating.
    """
    lower_bound: float
    upper_bound: float

    def get(self):
        return self

    def randomize(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, 2)

    def get_bounds(self):
        return [self.lower_bound] * 2, [self.upper_bound] * 2

    def correct(self, value):
        return [float(v) + 1 for v in value]

    def decode(self, value):
        return value

    def size(self):
        return 2

    def has_children(self):
        return False


def test_custom_variable():
    evaluated = []

    class Recorder(Task):
        def objective_function(self, x):
            evaluated.append(list(x))
            return float(np.sum(np.square(x)))

    task = Recorder(variables=[Rounded(name="r", lower_bound=0, upper_bound=1)], seed=0)
    # one value per dimension, also when randomize returns an array
    assert len(task.empty_solution()) == 2
    optimizer = GreyWolfOptimization(make_config())
    optimizer._task = task
    agent = optimizer._init_agent([0.0, 0.0])
    # the position is corrected once ([1, 1]), the objective is evaluated on the solution corrected again ([2, 2])
    assert agent.position == [1.0, 1.0]
    assert evaluated[-1] == [2.0, 2.0]


def test_firefly_never_worsens():
    # a firefly is replaced by its best candidate only if better: its cost never increases from a cycle to the next
    config = FireflySwarmOptimizationConfig(
        population_size=10, fitness_error=None, max_cycles=10, alpha=0.2, beta_min=2.0, gamma=0.001
    )
    result = FireflySwarmOptimization(config).optimize(make_task(seed=6))
    for before, after in zip(result.evolution, result.evolution[1:]):
        assert all(a.cost <= b.cost for b, a in zip(before.agents, after.agents))


def test_firefly_mutation_coefficient_is_damped():
    config = FireflySwarmOptimizationConfig(
        population_size=10, fitness_error=None, max_cycles=4, alpha=0.2, beta_min=2.0, gamma=0.001, alpha_damp=0.5
    )
    optimizer = FireflySwarmOptimization(config)
    optimizer.optimize(make_task(seed=6))
    assert optimizer._FireflySwarmOptimization__alpha == pytest.approx(0.2 * 0.5 ** 4)


def test_fish_school_keeps_the_weights():
    # the weight of a fish is gained by feeding and kept when it moves (it was reset at every move)
    config = FishSchoolSearchOptimizationConfig(
        population_size=10, fitness_error=None, max_cycles=5, step_individual_init=0.1, step_individual_final=0.0001,
        step_volitive_init=0.01, step_volitive_final=0.001, min_w=1, w_scale=500,
    )
    optimizer = FishSchoolSearchOptimization(config)
    optimizer.optimize(make_task(seed=2))
    assert any(fish.weight != config.w_scale / 2.0 for fish in optimizer._population)


def test_genetic_algorithm_children_and_odd_population():
    from pyvolutionary import GeneticAlgorithmOptimization, GeneticAlgorithmOptimizationConfig
    # the population keeps its (odd) size
    config = GeneticAlgorithmOptimizationConfig(population_size=7, fitness_error=None, max_cycles=3, px_over=0.9)
    result = GeneticAlgorithmOptimization(config).optimize(make_task(seed=1))
    assert all(len(population.agents) == 7 for population in result.evolution)

    # the two children of a pair differ, unless the same parent is selected twice (they were always identical)
    config = GeneticAlgorithmOptimizationConfig(population_size=20, fitness_error=None, max_cycles=1, px_over=0.999)
    optimizer = GeneticAlgorithmOptimization(config)
    optimizer.optimize(make_task(seed=1))
    genes = [gene.bitstring for gene in optimizer._GeneticAlgorithmOptimization__bit_genes]
    assert sum(genes[idx] == genes[idx + 1] for idx in range(0, 20, 2)) <= 3


def test_clusters_include_every_agent():
    from pyvolutionary.helpers import split_in_clusters
    agents = [Agent(position=[float(i)], cost=float(i), fitness=0.0) for i in range(8)]
    clusters = split_in_clusters(agents, 5)
    assert [len(c) for c in clusters] == [2, 2, 2, 1, 1]
    assert [a.cost for c in clusters for a in c] == [float(i) for i in range(8)]
    assert len(split_in_clusters(agents, 20)) == 8
