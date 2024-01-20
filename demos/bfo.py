from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import BacterialForagingOptimization, BacterialForagingOptimizationConfig
from pyvolutionary.utils import animate

configuration = BacterialForagingOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    C_s=0.1,
    C_e=0.001,
    Ped=0.01,
    Ns=4,
    N_adapt=2,
    N_split=40,
)

optimization_result = BacterialForagingOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/bfa_{name}.gif")
