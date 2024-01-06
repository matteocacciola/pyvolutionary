from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import CoyotesOptimization, CoyotesOptimizationConfig, animate

configuration = CoyotesOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    num_coyotes=3,
)

optimization_result = CoyotesOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/co_{name}.gif")
