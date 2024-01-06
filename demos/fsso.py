from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import FishSchoolSearchOptimization, FishSchoolSearchOptimizationConfig, animate

configuration = FishSchoolSearchOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    step_individual_init=0.1,
    step_individual_final=0.0001,
    step_volitive_init=0.01,
    step_volitive_final=0.001,
    min_w=1.0,
    w_scale=500.0,
)

optimization_result = FishSchoolSearchOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/fsso_{name}.gif")
