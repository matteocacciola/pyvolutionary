from pyvolutionary import OptimizationResult, AntColonyOptimization, AntColonyOptimizationConfig, EarlyStopping
from tests.fixtures import task as fixture_task

# the fixture task is shared with other tests: seed a copy, so that each run is reproducible (unseeded runs could get
# stuck in a local minimum of the Rastrigin function and last up to max_cycles)
task = fixture_task.model_copy(update={"seed": 0})


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
    # the optimization stops because the fitness error is reached, well before max_cycles
    assert result.rates[-1] <= 0.1
    assert len(result.evolution) < 1000


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
