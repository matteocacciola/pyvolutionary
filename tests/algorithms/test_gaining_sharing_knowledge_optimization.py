import pytest

from pyvolutionary import (
    OptimizationResult,
    GainingSharingKnowledgeOptimizationConfig,
    GainingSharingKnowledgeOptimization,
)
from tests.fixtures import task


@pytest.fixture
def optimization_config() -> GainingSharingKnowledgeOptimizationConfig:
    return GainingSharingKnowledgeOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        p=0.1,
        kf=0.5,
        kr=0.9,
        kg=5,
    )


def test_valid_optimization(optimization_config):
    o = GainingSharingKnowledgeOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = GainingSharingKnowledgeOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = GainingSharingKnowledgeOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
