from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import MonarchButterflyOptimization, MonarchButterflyOptimizationConfig, animate

configuration = MonarchButterflyOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    partition=5.0 / 12.0,
    period=1.2,
)

optimization_result = MonarchButterflyOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/mbo_{name}.gif")
