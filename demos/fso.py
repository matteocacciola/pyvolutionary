from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import FireflySwarmOptimization, FireflySwarmOptimizationConfig
from pyvolutionary.utils import animate

configuration = FireflySwarmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    alpha=0.5,
    beta_min=0.2,
    gamma=0.99,
)

optimization_result = FireflySwarmOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/fso_{name}.gif")
