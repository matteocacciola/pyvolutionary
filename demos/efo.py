from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import ElectromagneticFieldOptimization, ElectromagneticFieldOptimizationConfig, animate

configuration = ElectromagneticFieldOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    r_rate=0.3,
    ps_rate=0.85,
    p_field=0.1,
    n_field=0.45,
)

optimization_result = ElectromagneticFieldOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/efo_{name}.gif")
