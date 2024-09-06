from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import MultiverseOptimization, MultiverseOptimizationConfig

configuration = MultiverseOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    wep_min=0.2,
    wep_max=1.0,
)

optimization_result = MultiverseOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/mvo_{name}.gif")
