import pytest

from pyvolutionary import OptimizationResult, BiogeographyBasedOptimization, BiogeographyBasedOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> BiogeographyBasedOptimizationConfig:
    return BiogeographyBasedOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        p_m=0.2,
        n_elites=5,
    )


def test_valid_optimization(optimization_config):
    o = BiogeographyBasedOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = BiogeographyBasedOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = BiogeographyBasedOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
