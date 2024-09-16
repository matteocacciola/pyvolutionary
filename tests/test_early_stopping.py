from pyvolutionary import OptimizationResult, AntColonyOptimization, AntColonyOptimizationConfig, EarlyStopping
from tests.fixtures import task


def test_max_cycles():
    optimization_config = AntColonyOptimizationConfig(
        population_size=20,
        max_cycles=1,
        archive_size=20,
        intent_factor=0.1,
        zeta=0.85,
    )
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_fitness_error():
    optimization_config = AntColonyOptimizationConfig(
        population_size=20,
        fitness_error=0.1,
        max_cycles=1e5,
        archive_size=20,
        intent_factor=0.1,
        zeta=0.85,
    )
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_early_stopping_no_patience():
    optimization_config = AntColonyOptimizationConfig(
        population_size=20,
        fitness_error=0.0001,
        max_cycles=1e5,
        early_stopping=EarlyStopping(min_delta=0.01),
        archive_size=20,
        intent_factor=0.1,
        zeta=0.85,
    )
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_early_stopping_with_patience():
    optimization_config = AntColonyOptimizationConfig(
        population_size=20,
        fitness_error=0.0001,
        max_cycles=1e5,
        early_stopping=EarlyStopping(patience=3, min_delta=0.01),
        archive_size=20,
        intent_factor=0.1,
        zeta=0.85,
    )
    o = AntColonyOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)
