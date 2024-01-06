from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import TasmanianDevilOptimization, TasmanianDevilOptimizationConfig, animate

configuration = TasmanianDevilOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
)

optimization_result = TasmanianDevilOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/tdo_{name}.gif")
