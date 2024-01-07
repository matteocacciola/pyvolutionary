import pytest

from pyvolutionary import (
    OptimizationResult,
    GizaPyramidConstructionOptimization,
    GizaPyramidConstructionOptimizationConfig,
)
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> GizaPyramidConstructionOptimizationConfig:
    return GizaPyramidConstructionOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        theta=14,
        friction=[1, 10],
        prob_substitution=0.5,
    )


def test_valid_optimization(optimization_config):
    o = GizaPyramidConstructionOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = GizaPyramidConstructionOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = GizaPyramidConstructionOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
