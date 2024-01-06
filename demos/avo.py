from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import AfricanVultureOptimization, AfricanVultureOptimizationConfig, animate

configuration = AfricanVultureOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    p=[0.6, 0.4, 0.6],
    alpha=0.8,
    gamma=2.5,
)

optimization_result = AfricanVultureOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/avo_{name}.gif")
