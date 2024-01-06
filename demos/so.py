from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import SeagullOptimization, SeagullOptimizationConfig, animate

configuration = SeagullOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    fc=2,
)

optimization_result = SeagullOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/so_{name}.gif")
