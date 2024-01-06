import pytest

from pyvolutionary import OptimizationResult, ElectromagneticFieldOptimization, ElectromagneticFieldOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ElectromagneticFieldOptimizationConfig:
    return ElectromagneticFieldOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        r_rate=0.3,
        ps_rate=0.85,
        p_field=0.1,
        n_field=0.45,
    )


def test_valid_optimization(optimization_config):
    o = ElectromagneticFieldOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ElectromagneticFieldOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ElectromagneticFieldOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
