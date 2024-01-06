import pytest

from pyvolutionary import (
    OptimizationResult,
    ImperialistCompetitiveOptimization,
    ImperialistCompetitiveOptimizationConfig,
)
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ImperialistCompetitiveOptimizationConfig:
    return ImperialistCompetitiveOptimizationConfig(
        population_size=20,
        max_cycles=30,
        fitness_error=0.01,
        assimilation_rate=0.4,
        revolution_rate=0.1,
        alpha_rate=0.8,
        revolution_probability=0.2,
        number_of_countries=300,
    )


def test_valid_optimization(optimization_config):
    o = ImperialistCompetitiveOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ImperialistCompetitiveOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ImperialistCompetitiveOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
