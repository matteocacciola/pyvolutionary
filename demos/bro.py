from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import BattleRoyaleOptimization, BattleRoyaleOptimizationConfig

configuration = BattleRoyaleOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    threshold=3,
)

optimization_result = BattleRoyaleOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/bro_{name}.gif")
