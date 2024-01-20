from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import BiogeographyBasedOptimization, BiogeographyBasedOptimizationConfig
from pyvolutionary.utils import animate

configuration = BiogeographyBasedOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    p_m=0.2,
    n_elites=5,
)

optimization_result = BiogeographyBasedOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/bgbo_{name}.gif")
