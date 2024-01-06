from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import HarmonySearchOptimization, HarmonySearchOptimizationConfig, animate

configuration = HarmonySearchOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    consideration_rate=0.15,
    pitch_adjusting_rate=0.5,
)

optimization_result = HarmonySearchOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/hso_{name}.gif")
