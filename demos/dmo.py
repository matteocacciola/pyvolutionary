from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from demos.utils.utils import animate
from pyvolutionary import DwarfMongooseOptimization, DwarfMongooseOptimizationConfig

configuration = DwarfMongooseOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    n_baby_sitter=3,
    peep=2,
)

optimization_result = DwarfMongooseOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/dmo_{name}.gif")
