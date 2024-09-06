from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import CoronavirusHerdImmunityOptimization, CoronavirusHerdImmunityOptimizationConfig

configuration = CoronavirusHerdImmunityOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    C0=0.1,
    brr=0.05,
    max_age=10,
)

optimization_result = CoronavirusHerdImmunityOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/chio_{name}.gif")
