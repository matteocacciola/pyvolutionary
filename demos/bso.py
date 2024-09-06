from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import BrainStormOptimization, BrainStormOptimizationConfig

configuration = BrainStormOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    m_clusters=5,
    p1=0.2,
    p2=0.8,
    p3=0.4,
    p4=0.5,
    slope=20,
)

optimization_result = BrainStormOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/bso_{name}.gif")
