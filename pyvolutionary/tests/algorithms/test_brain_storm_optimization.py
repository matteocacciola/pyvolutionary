import pytest

from pyvolutionary import OptimizationResult, BrainStormOptimization, BrainStormOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> BrainStormOptimizationConfig:
    return BrainStormOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        m_clusters=5,
        p1=0.2,
        p2=0.8,
        p3=0.4,
        p4=0.5,
        slope=20,
    )


def test_valid_optimization(optimization_config):
    o = BrainStormOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = BrainStormOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = BrainStormOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
