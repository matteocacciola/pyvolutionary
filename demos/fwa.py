from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import FireworksOptimization, FireworksOptimizationConfig

configuration = FireworksOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    sparks_num=50,
    a=0.04,
    b=0.8,
    explosion_amplitude=40,
    gaussian_explosion_number=5,
)

optimization_result = FireworksOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/fwa_{name}.gif")
