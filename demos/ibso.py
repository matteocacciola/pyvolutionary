from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import ImprovedBrainStormOptimization, ImprovedBrainStormOptimizationConfig
from pyvolutionary.utils import animate

configuration = ImprovedBrainStormOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    m_clusters=5,
    p1=0.2,
    p2=0.8,
    p3=0.4,
    p4=0.5,
)

optimization_result = ImprovedBrainStormOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/ibso_{name}.gif")
