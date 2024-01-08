import pytest

from pyvolutionary import OptimizationResult, CatSwarmOptimization, CatSwarmOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> CatSwarmOptimizationConfig:
    return CatSwarmOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        mixture_ratio=0.5,
        smp=10,
        cdc=0.5,
        srd=0.5,
        c1=2.0,
        w=[0.2, 0.9],
    )


def test_valid_optimization(optimization_config):
    for spc in [True, False]:
        optimization_config.spc = spc
        for s in [0, 1, 2, 3]:
            optimization_config.selected_strategy = s
            o = CatSwarmOptimization(optimization_config)
            result = o.optimize(task)
            assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    for spc in [True, False]:
        optimization_config.spc = spc
        for s in [0, 1, 2, 3]:
            optimization_config.selected_strategy = s
            o = CatSwarmOptimization(optimization_config)
            result = o.optimize(task, mode="thread")
            assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    for spc in [True, False]:
        optimization_config.spc = spc
        for s in [0, 1, 2, 3]:
            optimization_config.selected_strategy = s
            o = CatSwarmOptimization(optimization_config)
            result = o.optimize(task, mode="process")
            assert isinstance(result, OptimizationResult)
