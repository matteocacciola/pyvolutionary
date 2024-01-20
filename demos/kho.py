from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import KrillHerdOptimization, KrillHerdOptimizationConfig
from pyvolutionary.utils import animate

configuration = KrillHerdOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    n_max=0.01,
    foraging_speed=0.01,
    diffusion_speed=0.01,
    c_t=0.1,
    w_neighbour=0.5,
    w_foraging=0.5,
    max_neighbours=5,
    crossover_rate=0.7,
    mutation_rate=0.5,
)

optimization_result = KrillHerdOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/kho_{name}.gif")
