from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import AntColonyOptimization, AntColonyOptimizationConfig
from pyvolutionary.utils import animate

configuration = AntColonyOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    archive_size=20,
    intent_factor=0.1,
    zeta=0.85,
)

optimization_result = AntColonyOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/aco_{name}.gif")
