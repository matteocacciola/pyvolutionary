from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import CoralReefOptimization, CoralReefOptimizationConfig
from pyvolutionary.utils import animate

configuration = CoralReefOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    po=0.3,
    Fb=0.8,
    Fa=0.1,
    Fd=0.1,
    Pd=0.3,
    GCR=0.1,
    gamma=[0.01, 0.5],
    n_trials=5,
)

optimization_result = CoralReefOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/cro_{name}.gif")
