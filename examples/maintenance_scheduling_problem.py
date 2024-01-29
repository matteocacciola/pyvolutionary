# In maintenance scheduling, the goal is to optimize the schedule for performing maintenance tasks on various assets
# or equipment. The objective is to minimize downtime and maximize the utilization of assets while considering
# various constraints such as resource availability, task dependencies, and time constraints.
#
# Each element in the solution represents whether a task is assigned to an asset (1) or not (0). The schedule
# specifies when each task should start and which asset it is assigned to, aiming to minimize the total downtime.
#
# By using the Mealpy, you can find an efficient maintenance schedule that minimizes downtime, maximizes asset
# utilization, and satisfies various constraints, ultimately optimizing the maintenance operations for improved
# reliability and productivity.

import numpy as np
from pyvolutionary import BinaryVariable, Task, EgretSwarmOptimization, EgretSwarmOptimizationConfig

num_tasks = 10
num_assets = 5
task_durations = np.random.randint(1, 10, size=(num_tasks, num_assets))

data = {
    "num_tasks": num_tasks,
    "num_assets": num_assets,
    "task_durations": task_durations,
    "unassigned_penalty": -100  # Define a penalty value for no task is assigned to asset
}


class MaintenanceSchedulingProblem(Task):
    def objective_function(self, x):
        x_decoded = np.array(self.transform_solution(x)["task_var"])
        downtime = -np.sum(x_decoded.reshape(
            (self.data["num_tasks"], self.data["num_assets"])
        ) * self.data["task_durations"])
        if np.sum(x_decoded) == 0:
            downtime += self.data["unassigned_penalty"]
        return downtime


problem = MaintenanceSchedulingProblem(
    variables=[BinaryVariable(n_vars=num_tasks * num_assets, name="task_var")],
    minmax="max",
    data=data,
)

config = EgretSwarmOptimizationConfig(population_size=20, fitness_error=0.1, max_cycles=100)
result = EgretSwarmOptimization(config).optimize(problem)

best_scheduling = np.array(problem.transform_solution(result.best_solution.position)["task_var"]).reshape(
    (num_tasks, num_assets)
)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {problem.transform_solution(result.best_solution.position)}")  # Decoded (Real) solution