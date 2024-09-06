from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import CamelCaravanOptimization, CamelCaravanOptimizationConfig

configuration = CamelCaravanOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    burden_factor=0.5,
    death_rate=0.5,
    visibility=0.5,
    supply=10,
    endurance=10,
    temperatures=[-10, 10],
)

optimization_result = CamelCaravanOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/ccoa_{name}.gif")
