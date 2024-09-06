from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import SwarmHillClimbingOptimization, SwarmHillClimbingOptimizationConfig

configuration = SwarmHillClimbingOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    neighbour_size=50,
)

optimization_result = SwarmHillClimbingOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/shco_{name}.gif")
