from opfunu.cec_based.cec2017 import F52017

from pyvolutionary import Task, BiogeographyBasedOptimization, HyperTuner, ContinuousMultiVariable

f1 = F52017(30, f_bias=0)


class Problem(Task):
    # Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    def objective_function(self, solution):
        return f1.evaluate(solution)


# Define the task with the bounds and the configuration of the optimizer
task = Problem(
    variables=[ContinuousMultiVariable(name="x", lower_bounds=f1.lb, upper_bounds=f1.ub)],
)

params_bbo_grid = {
    "max_cycles": [10, 20, 30, 40],
    "population_size": [50, 100, 150],
    "n_elites": [3, 4, 5, 6],
    "p_m": [0.01, 0.02, 0.05]
}


model = BiogeographyBasedOptimization()
tuner = HyperTuner(model, params_bbo_grid)

tuner.execute(task=task, debug=True, n_trials=2, n_jobs=2)

print(f"Best row {tuner.best_row}")
print(f"Best score {tuner.best_score}")
print(f"Best parameters {tuner.best_parameters}")

best_result = tuner.resolve()
print(f"Best solution after tuning {best_result.best_solution}")

tuner.export_results("csv")
tuner.export_results("dataframe")
tuner.export_results("json")
