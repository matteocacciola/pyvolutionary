from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import WaterCycleOptimization, WaterCycleOptimizationConfig

configuration = WaterCycleOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    nsr=4,
    wc=2.0,
)

optimization_result = WaterCycleOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/wco_{name}.gif")
