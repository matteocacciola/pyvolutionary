from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import CatSwarmOptimization, CatSwarmOptimizationConfig
from pyvolutionary.utils import animate

configuration = CatSwarmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    mixture_ratio=0.5,
    smp=10,
    cdc=0.5,
    srd=0.5,
    c1=0.1,
    w=[0.35, 1],
    spc=True,
    selected_strategy=0,
)

optimization_result = CatSwarmOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/cso_{name}.gif")
