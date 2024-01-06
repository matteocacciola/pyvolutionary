from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import BatOptimization, BatOptimizationConfig, animate

configuration = BatOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    loudness=[1, 2],
    pulse_rate=[0.15, 0.85],
    pulse_frequency=[-10, 10],
)

optimization_result = BatOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/bo_{name}.gif")
