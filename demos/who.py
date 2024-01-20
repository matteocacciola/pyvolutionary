from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import WildebeestHerdOptimization, WildebeestHerdOptimizationConfig
from pyvolutionary.utils import animate

configuration = WildebeestHerdOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    n_explore_step=2,
    n_exploit_step=2,
    eta=0.1,
    phi=0.1,
    local_alpha=0.5,
    local_beta=0.5,
    global_alpha=0.5,
    global_beta=0.5,
    delta_w=1.0,
    delta_c=1.0,
)

optimization_result = WildebeestHerdOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/who_{name}.gif")
