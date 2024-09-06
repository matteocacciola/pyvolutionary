from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import ElephantHerdOptimization, ElephantHerdOptimizationConfig

configuration = ElephantHerdOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    alpha=0.5,
    beta=0.5,
    n_clans=3,
)

optimization_result = ElephantHerdOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/eho_{name}.gif")
