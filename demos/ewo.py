from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import EarthwormsOptimization, EarthwormsOptimizationConfig
from pyvolutionary.utils import animate

configuration = EarthwormsOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    prob_mutate=0.01,
    prob_crossover=0.8,
    keep=5,
    alpha=0.98,
    beta=0.95,
    gamma=0.9,
)

optimization_result = EarthwormsOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/ewo_{name}.gif")
