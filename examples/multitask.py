from opfunu.cec_based.cec2017 import F52017, F102017, F292017

from pyvolutionary import (
    ContinuousVariable,
    Task,
    NuclearReactionOptimization,
    Multitask,
    NuclearReactionOptimizationConfig,
    MountainGazelleOptimization,
    MountainGazelleOptimizationConfig,
    GrasshopperOptimization,
    GrasshopperOptimizationConfig,
    GizaPyramidConstructionOptimization,
    GizaPyramidConstructionOptimizationConfig,
)

f1 = F52017(30, f_bias=0)
f2 = F102017(30, f_bias=0)
f3 = F292017(30, f_bias=0)


class Problem1(Task):
    def objective_function(self, solution):
        return f1.evaluate(solution)


class Problem2(Task):
    def objective_function(self, solution):
        return f3.evaluate(solution)


class Problem3(Task):
    def objective_function(self, solution):
        return f1.evaluate(solution)


task1 = Problem1(
    variables=[
        ContinuousVariable(name=f"x{i}", lower_bound=f1.lb[i], upper_bound=f1.ub[i]) for i in range(0, f1.ndim)
    ],
)
task2 = Problem2(
    variables=[
        ContinuousVariable(name=f"x{i}", lower_bound=f2.lb[i], upper_bound=f2.ub[i]) for i in range(0, f2.ndim)
    ],
)
task3 = Problem3(
    variables=[
        ContinuousVariable(name=f"x{i}", lower_bound=f3.lb[i], upper_bound=f3.ub[i]) for i in range(0, f3.ndim)
    ],
)

model1 = NuclearReactionOptimization(
    config=NuclearReactionOptimizationConfig(max_cycles=10000, population_size=50)
)
model2 = MountainGazelleOptimization(
    config=MountainGazelleOptimizationConfig(max_cycles=10000, population_size=50)
)
model3 = GrasshopperOptimization(
    config=GrasshopperOptimizationConfig(max_cycles=10000, population_size=50, c_min=0.00004, c_max=2.0,)
)
model4 = GizaPyramidConstructionOptimization(
    config=GizaPyramidConstructionOptimizationConfig(
        max_cycles=10000, population_size=50, theta=14, friction=[1, 10], prob_substitution=0.5,
    )
)

multitask = Multitask(
    algorithms=(model1, model2, model3, model4), tasks=(task1, task2, task3), modes=("thread", ), n_workers=4
)

multitask.execute(n_trials=2, n_jobs=2, debug=True)

multitask.export_results("csv")
multitask.export_results("dataframe")
multitask.export_results("json")
