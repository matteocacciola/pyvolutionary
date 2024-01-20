import pytest

from pyvolutionary import OptimizationResult, KrillHerdOptimization, KrillHerdOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> KrillHerdOptimizationConfig:
    return KrillHerdOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        n_max=0.01,
        foraging_speed=0.01,
        diffusion_speed=0.01,
        c_t=0.01,
        w_neighbour=0.01,
        w_foraging=0.01,
        max_neighbours=5,
        crossover_rate=0.01,
        mutation_rate=0.01,
    )


def test_valid_optimization(optimization_config):
    o = KrillHerdOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = KrillHerdOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = KrillHerdOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
