from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import HenryGasSolubilityOptimization, HenryGasSolubilityOptimizationConfig

configuration = HenryGasSolubilityOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    n_clusters=3,
)

optimization_result = HenryGasSolubilityOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/hgsop_{name}.gif")
