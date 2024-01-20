from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import GrasshopperOptimization, GrasshopperOptimizationConfig
from pyvolutionary.utils import animate

configuration = GrasshopperOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    c_min=0.00004,
    c_max=2.0,
)

optimization_result = GrasshopperOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/goa_{name}.gif")
