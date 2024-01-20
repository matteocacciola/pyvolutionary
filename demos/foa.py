from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import ForestOptimizationAlgorithm, ForestOptimizationAlgorithmConfig
from pyvolutionary.utils import animate

configuration = ForestOptimizationAlgorithmConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    lifetime=5,
    area_limit=50,
    local_seeding_changes=1,
    global_seeding_changes=2,
    transfer_rate=0.5,
)

optimization_result = ForestOptimizationAlgorithm(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/foa_{name}.gif")
