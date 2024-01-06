from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import VirusColonySearchOptimization, VirusColonySearchOptimizationConfig, animate

configuration = VirusColonySearchOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    lamda=0.1,
    sigma=2.5,
)

optimization_result = VirusColonySearchOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/vcso_{name}.gif")
