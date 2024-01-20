from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import GizaPyramidConstructionOptimization, GizaPyramidConstructionOptimizationConfig
from pyvolutionary.utils import animate

configuration = GizaPyramidConstructionOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    theta=14,
    friction=[1, 10],
    prob_substitution=0.5,
)

optimization_result = GizaPyramidConstructionOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/gpco_{name}.gif")
