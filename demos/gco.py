from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import GerminalCenterOptimization, GerminalCenterOptimizationConfig
from pyvolutionary.utils import animate

configuration = GerminalCenterOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    cr=0.7,
    wf=1.25,
)

optimization_result = GerminalCenterOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/gco_{name}.gif")
