from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import QleSineCosineAlgorithmOptimization, QleSineCosineAlgorithmOptimizationConfig

configuration = QleSineCosineAlgorithmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    alpha=0.1,
    gama=0.9,
)

optimization_result = QleSineCosineAlgorithmOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/qlescao_{name}.gif")
