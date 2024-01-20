from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import ArchimedeOptimization, ArchimedeOptimizationConfig
from pyvolutionary.utils import animate

configuration = ArchimedeOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    c1=2.0,
    c2=2.0,
    c3=2.0,
    c4=0.5,
    acc=[0.2, 0.9],
)

optimization_result = ArchimedeOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/archo_{name}.gif")
