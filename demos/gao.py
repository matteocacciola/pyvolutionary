from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import GeneticAlgorithmOptimization, GeneticAlgorithmOptimizationConfig, animate

configuration = GeneticAlgorithmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    px_over=0.8,
)

optimization_result = GeneticAlgorithmOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/gao_{name}.gif")
