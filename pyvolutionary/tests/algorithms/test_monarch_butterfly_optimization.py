import pytest

from pyvolutionary import OptimizationResult, MonarchButterflyOptimization, MonarchButterflyOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> MonarchButterflyOptimizationConfig:
    return MonarchButterflyOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        partition=5.0 / 12.0,
        period=1.2,
    )


def test_valid_optimization(optimization_config):
    o = MonarchButterflyOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = MonarchButterflyOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = MonarchButterflyOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
