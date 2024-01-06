from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import AquilaOptimization, AquilaOptimizationConfig, animate

configuration = AquilaOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
)

optimization_result = AquilaOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/ao_{name}.gif")
