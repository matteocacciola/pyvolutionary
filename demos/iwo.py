from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import InvasiveWeedOptimization, InvasiveWeedOptimizationConfig

configuration = InvasiveWeedOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    seed=[1, 4],
    exponent=2,
    sigma=[0.5, 0.1],
)

optimization_result = InvasiveWeedOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/iwo_{name}.gif")
