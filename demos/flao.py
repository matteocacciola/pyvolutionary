from demos.functions.sphere import (
    fitness_error, task, generation, name, position_min, position_max, population,
)
from pyvolutionary import FicksLawOptimization, FicksLawOptimizationConfig
from pyvolutionary.utils import animate

configuration = FicksLawOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    C1=0.5,
    C2=2.0,
    C3=0.1,
    C4=0.2,
    C5=2.0,
    DD=0.01,
)

optimization_result = FicksLawOptimization(configuration, True).optimize(task)
animate(task.objective_function, optimization_result, position_min, position_max, f"demos/images/flao_{name}.gif")
