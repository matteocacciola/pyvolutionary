from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import ImperialistCompetitiveOptimization, ImperialistCompetitiveOptimizationConfig
from pyvolutionary.utils import animate

configuration = ImperialistCompetitiveOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    assimilation_rate=0.4,
    revolution_rate=0.1,
    alpha_rate=0.8,
    revolution_probability=0.2,
    number_of_countries=300,
)

optimization_result = ImperialistCompetitiveOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/ico_{name}.gif")
