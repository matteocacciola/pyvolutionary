import pytest

from pyvolutionary import OptimizationResult, ImprovedBrainStormOptimization, ImprovedBrainStormOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ImprovedBrainStormOptimizationConfig:
    return ImprovedBrainStormOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        m_clusters=5,
        p1=0.2,
        p2=0.8,
        p3=0.4,
        p4=0.5,
    )


def test_valid_optimization(optimization_config):
    o = ImprovedBrainStormOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ImprovedBrainStormOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ImprovedBrainStormOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
