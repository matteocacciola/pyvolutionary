from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import MarinePredatorsOptimization, MarinePredatorsOptimizationConfig
from pyvolutionary.utils import animate

configuration = MarinePredatorsOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
)

optimization_result = MarinePredatorsOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/mpo_{name}.gif")
