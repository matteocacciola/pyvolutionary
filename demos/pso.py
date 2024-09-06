from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import ParticleSwarmOptimization, ParticleSwarmOptimizationConfig

configuration = ParticleSwarmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    c1=0.1,
    c2=0.1,
    w=[0.35, 1],
)

optimization_result = ParticleSwarmOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/pso_{name}.gif")
