from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import BeeColonyOptimization, BeeColonyOptimizationConfig

configuration = BeeColonyOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    scouting_limit=5,
)

optimization_result = BeeColonyOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/bco_{name}.gif")
