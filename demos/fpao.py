from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import FlowerPollinationAlgorithmOptimization, FlowerPollinationAlgorithmOptimizationConfig, animate

configuration = FlowerPollinationAlgorithmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    p_s=0.8,
    levy_multiplier=0.2,
)

optimization_result = FlowerPollinationAlgorithmOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/fpao_{name}.gif")
