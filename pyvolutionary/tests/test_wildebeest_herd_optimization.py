import pytest

from pyvolutionary import OptimizationResult, WildebeestHerdOptimization, WildebeestHerdOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> WildebeestHerdOptimizationConfig:
    return WildebeestHerdOptimizationConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        n_explore_step=2,
        n_exploit_step=2,
        eta=0.1,
        phi=0.1,
        local_alpha=0.5,
        local_beta=0.5,
        global_alpha=0.5,
        global_beta=0.5,
        delta_w=1.0,
        delta_c=1.0,
    )


def test_valid_optimization(optimization_config):
    o = WildebeestHerdOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = WildebeestHerdOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = WildebeestHerdOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
