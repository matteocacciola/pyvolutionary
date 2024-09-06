from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import HungerGamesSearchOptimization, HungerGamesSearchOptimizationConfig

configuration = HungerGamesSearchOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    PUP=0.08,
    LH=10000,
)

optimization_result = HungerGamesSearchOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/hgso_{name}.gif")
