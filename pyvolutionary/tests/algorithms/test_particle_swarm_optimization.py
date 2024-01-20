import pytest

from pyvolutionary import OptimizationResult, ParticleSwarmOptimization, ParticleSwarmOptimizationConfig
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ParticleSwarmOptimizationConfig:
    return ParticleSwarmOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        c1=2.0,
        c2=2.0,
        w=[0.2, 0.9],
    )


def test_valid_optimization(optimization_config):
    o = ParticleSwarmOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(optimization_config):
    o = ParticleSwarmOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(optimization_config):
    o = ParticleSwarmOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)
