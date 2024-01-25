from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import ForensicBasedInvestigationOptimization, ForensicBasedInvestigationOptimizationConfig
from pyvolutionary.utils import animate

configuration = ForensicBasedInvestigationOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
)

optimization_result = ForensicBasedInvestigationOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/fbio_{name}.gif")
