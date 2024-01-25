from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import GiantTrevallyOptimization, GiantTrevallyOptimizationConfig
from pyvolutionary.utils import animate

configuration = GiantTrevallyOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
)

optimization_result = GiantTrevallyOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/gto_{name}.gif")
