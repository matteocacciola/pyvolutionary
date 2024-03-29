import pytest

from pyvolutionary import (
    OptimizationResult,
    QleSineCosineAlgorithmOptimization,
    QleSineCosineAlgorithmOptimizationConfig,
)
from tests.fixtures import task


@pytest.fixture
def optimization_config() -> QleSineCosineAlgorithmOptimizationConfig:
    return QleSineCosineAlgorithmOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=10,
        alpha=0.1,
        gama=0.9,
    )


def test_valid_optimization(optimization_config):
    o = QleSineCosineAlgorithmOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = QleSineCosineAlgorithmOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = QleSineCosineAlgorithmOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
