from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import CuckooSearchOptimization, CuckooSearchOptimizationConfig, animate

configuration = CuckooSearchOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    p_a=0.5,
)

optimization_result = CuckooSearchOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/csoa_{name}.gif")
