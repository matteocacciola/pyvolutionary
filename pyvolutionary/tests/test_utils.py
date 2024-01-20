import numpy as np
import pytest

from pyvolutionary import (
    ZebraOptimizationConfig,
    ZebraOptimization,
    agent_trend,
    best_agent_trend,
    agent_position,
    best_agent_position,
)
from pyvolutionary.tests.fixtures import task


@pytest.fixture
def optimization_config() -> ZebraOptimizationConfig:
    return ZebraOptimizationConfig(population_size=20, fitness_error=0.01, max_cycles=100)


def test_valid_agent_trend_results(optimization_config):
    zoa = ZebraOptimization(optimization_config)
    result = zoa.optimize(task)

    max_iters = len(result.evolution)

    idx = np.random.randint(0, optimization_config.population_size)
    trend = agent_trend(result, idx)
    assert len(trend) == max_iters

    num_iters_pick = np.random.randint(0, max_iters)
    partial_trend = agent_trend(
        result, idx, iters=np.random.randint(0, max_iters, num_iters_pick).tolist()
    )
    assert len(partial_trend) == num_iters_pick


def test_valid_best_agent_trend_results(optimization_config):
    zoa = ZebraOptimization(optimization_config)
    result = zoa.optimize(task)

    max_iters = len(result.evolution)

    trend = best_agent_trend(result)
    assert len(trend) == max_iters

    num_iters_pick = np.random.randint(0, max_iters)
    partial_trend = best_agent_trend(
        result, iters=np.random.randint(0, max_iters, num_iters_pick).tolist()
    )
    assert len(partial_trend) == num_iters_pick


def test_valid_agent_position_results(optimization_config):
    zoa = ZebraOptimization(optimization_config)
    result = zoa.optimize(task)

    max_iters = len(result.evolution)

    idx = np.random.randint(0, optimization_config.population_size)
    positions = agent_position(result, idx)
    assert len(positions) == max_iters

    num_iters_pick = np.random.randint(0, max_iters)
    partial_positions = agent_position(
        result, idx, iters=np.random.randint(0, max_iters, num_iters_pick).tolist()
    )
    assert len(partial_positions) == num_iters_pick


def test_valid_best_agent_position_results(optimization_config):
    zoa = ZebraOptimization(optimization_config)
    result = zoa.optimize(task)

    max_iters = len(result.evolution)

    positions = best_agent_trend(result)
    assert len(positions) == max_iters

    num_iters_pick = np.random.randint(0, max_iters)
    partial_positions = best_agent_position(
        result, iters=np.random.randint(0, max_iters, num_iters_pick).tolist()
    )
    assert len(partial_positions) == num_iters_pick
