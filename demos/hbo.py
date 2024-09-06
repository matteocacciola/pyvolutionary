from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import HeapBasedOptimization, HeapBasedOptimizationConfig

configuration = HeapBasedOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    degree=2,
)

optimization_result = HeapBasedOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/hbo_{name}.gif")
