# Let's consider a simplified example of production optimization in the context of a manufacturing company that
# produces electronic devices, such as smartphones. The objective is to maximize production output while minimizing
# production costs.
#
# This example uses binary representations for production configurations, assuming each task can be assigned to a
# resource (1) or not (0). You may need to adapt the representation and operators to suit your specific production
# optimization problem.

import numpy as np
from pyvolutionary import BinaryVariable, Task, FireflySwarmOptimization, FireflySwarmOptimizationConfig

# Define the problem parameters
num_tasks = 10
num_resources = 5

# Example task processing times
task_processing_times = np.array([2, 3, 4, 2, 3, 2, 3, 4, 2, 3])

# Example resource capacity
resource_capacity = np.array([10, 8, 6, 12, 15])

# Example production costs and outputs
production_costs = np.array([5, 6, 4, 7, 8, 9, 5, 6, 7, 8])
production_outputs = np.array([20, 18, 16, 22, 25, 24, 20, 18, 19, 21])

# Example maximum total production time
max_total_time = 50

# Example maximum defect rate
max_defect_rate = 0.2

# Penalty for invalid solution
penalty = -1000

data = {
    "num_tasks": num_tasks,
    "num_resources": num_resources,
    "task_processing_times": task_processing_times,
    "resource_capacity": resource_capacity,
    "production_costs": production_costs,
    "production_outputs": production_outputs,
    "max_defect_rate": max_defect_rate,
    "penalty": penalty
}


class SupplyChainProblem(Task):
    def objective_function(self, x):
        x_transformed = np.array(self.transform_solution(x)["placement_var"])
        x_transformed = x_transformed.reshape((self.data["num_tasks"], self.data["num_resources"]))

        # If any row has all 0 value, it indicates that this task is not allocated to any resource
        if np.any(np.all(x_transformed == 0, axis=1)) or np.any(np.all(x_transformed == 0, axis=0)):
            return self.data["penalty"]

        # Check violated constraints
        violated_constraints = 0

        # Calculate resource utilization
        resource_utilization = np.sum(x_transformed, axis=0)
        # Resource capacity constraint
        if np.any(resource_utilization > self.data["resource_capacity"]):
            violated_constraints += 1

        # Time constraint
        total_time = np.sum(np.dot(self.data["task_processing_times"].reshape(1, -1), x_transformed))
        if total_time > max_total_time:
            violated_constraints += 1

        # Quality constraint
        defect_rate = np.dot(
            self.data["production_costs"].reshape(1, -1), x_transformed
        ) / np.dot(self.data["production_outputs"], x_transformed)
        if np.any(defect_rate > max_defect_rate):
            violated_constraints += 1

        # Calculate the fitness value based on the objectives and constraints
        profit = np.sum(np.dot(self.data["production_outputs"].reshape(1, -1), x_transformed)) - np.sum(
            np.dot(self.data["production_costs"].reshape(1, -1), x_transformed)
        )
        if violated_constraints > 0:
            return profit + self.data["penalty"] * violated_constraints  # Penalize solutions with violated constraints
        return profit


problem = SupplyChainProblem(
    variables=[BinaryVariable(n_vars=num_tasks * num_resources, name="placement_var")],
    minmax="max",
    data=data
)

config = FireflySwarmOptimizationConfig(
    population_size=20,
    fitness_error=0.1,
    max_cycles=50,
    alpha=0.5,
    beta_min=0.2,
    gamma=0.99,
)
result = FireflySwarmOptimization(config).optimize(problem)

best_scheduling = np.array(problem.transform_solution(result.best_solution.position)["placement_var"]).reshape(
    (num_tasks, num_resources)
)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {best_scheduling}")  # Decoded (Real) solution
