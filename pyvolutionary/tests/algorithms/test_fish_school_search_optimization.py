import pytest

from pyvolutionary import OptimizationResult, FishSchoolSearchOptimization, FishSchoolSearchOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> FishSchoolSearchOptimizationConfig:
    return FishSchoolSearchOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        step_individual_init=0.1,
        step_individual_final=0.0001,
        step_volitive_init=0.01,
        step_volitive_final=0.001,
        min_w=1.0,
        w_scale=500.0,
    )


def test_valid_optimization(optimization_config):
    o = FishSchoolSearchOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = FishSchoolSearchOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = FishSchoolSearchOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
