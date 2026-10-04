from pyvolutionary import GreyWolfOptimization, GreyWolfOptimizationConfig
from tests.fixtures import Rastrigin, lower_bounds, upper_bounds
from pyvolutionary.models import ContinuousMultiVariable


def _optimize(seed):
    task = Rastrigin(
        variables=[ContinuousMultiVariable(name="x", lower_bounds=lower_bounds, upper_bounds=upper_bounds)],
        seed=seed,
    )
    config = GreyWolfOptimizationConfig(population_size=10, max_cycles=5)
    return GreyWolfOptimization(config).optimize(task).best_solution


def test_seed_makes_the_optimization_reproducible():
    # numpy>=2 refuses a float seed in np.random.seed: the seed must be kept as an integer
    first, second = _optimize(42), _optimize(42)
    assert first.position == second.position
    assert first.cost == second.cost


def test_integral_float_seed_is_accepted():
    assert _optimize(42.0).position == _optimize(42).position
