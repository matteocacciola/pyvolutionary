import pytest

from pyvolutionary import OptimizationResult, BacterialForagingOptimization, BacterialForagingOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> BacterialForagingOptimizationConfig:
    return BacterialForagingOptimizationConfig(
        population_size=10,
        fitness_error=0.01,
        max_cycles=100,
        C_s=0.1,
        C_e=0.001,
        Ped=0.01,
        Ns=4,
        N_adapt=2,
        N_split=40,
    )


def test_valid_optimization(optimization_config):
    o = BacterialForagingOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = BacterialForagingOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = BacterialForagingOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
