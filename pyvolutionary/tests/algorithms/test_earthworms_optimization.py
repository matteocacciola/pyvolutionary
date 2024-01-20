import pytest

from pyvolutionary import OptimizationResult, EarthwormsOptimization, EarthwormsOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> EarthwormsOptimizationConfig:
    return EarthwormsOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        prob_mutate=0.01,
        prob_crossover=0.8,
        keep=5,
        alpha=0.98,
        beta=0.95,
        gamma=0.9,
    )


def test_valid_optimization(optimization_config):
    o = EarthwormsOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = EarthwormsOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = EarthwormsOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
