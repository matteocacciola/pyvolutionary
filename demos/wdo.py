from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import WindDrivenOptimization, WindDrivenOptimizationConfig, animate

configuration = WindDrivenOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    RT=2,
    g_c=0.2,
    alp=0.4,
    c_e=0.5,
    max_v=0.3,
)

optimization_result = WindDrivenOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/wdo_{name}.gif")
