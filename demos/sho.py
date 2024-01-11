from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import SpottedHyenaOptimization, SpottedHyenaOptimizationConfig, animate

configuration = SpottedHyenaOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    h_factor=5.0,
    n_trials=10,
)

optimization_result = SpottedHyenaOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/sho_{name}.gif")
