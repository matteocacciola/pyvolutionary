from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import GainingSharingKnowledgeOptimization, GainingSharingKnowledgeOptimizationConfig

configuration = GainingSharingKnowledgeOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    p=0.1,
    kf=0.5,
    kr=0.9,
    kg=5,
)

optimization_result = GainingSharingKnowledgeOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/gsko_{name}.gif")
