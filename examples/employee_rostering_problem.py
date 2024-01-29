# The goal is to create an optimal schedule that assigns employees to shifts while satisfying various constraints and
# objectives. Note that this implementation assumes that shift_requirements array has dimensions (num_employees,
# num_shifts), and shift_costs is a 1D array of length num_shifts.
#
# Please keep in mind that this is a simplified implementation, and you may need to modify it according to the
# specific requirements and constraints of your employee rostering problem. Additionally, you might want to introduce
# additional mechanisms or constraints such as fairness, employee preferences, or shift dependencies to enhance the
# model's effectiveness in real-world scenarios.
#
# For example, if you have 5 employees and 3 shifts, a chromosome could be represented as [2, 1, 0, 2, 0],
# where employee 0 is assigned to shift 2, employee 1 is assigned to shift 1, employee 2 is assigned to shift 0,
# and so on.

import numpy as np
from pyvolutionary import DiscreteVariable, Task, FireflySwarmOptimization, FireflySwarmOptimizationConfig

shift_requirements = np.array([[2, 1, 3], [4, 2, 1], [3, 3, 2]])
shift_costs = np.array([10, 8, 12])

num_employees = shift_requirements.shape[0]
num_shifts = shift_requirements.shape[1]

data = {
    "shift_requirements": shift_requirements,
    "shift_costs": shift_costs,
    "num_employees": num_employees,
    "num_shifts": num_shifts
}


class EmployeeRosteringProblem(Task):
    def objective_function(self, x):
        x_decoded = self.transform_solution(x)
        x_decoded = [x_decoded[f"shift_var_{i}"] for i in range(self.data["num_employees"])]
        shifts_covered = np.zeros(self.data["num_shifts"])
        total_cost = 0
        for idx in range(self.data["num_employees"]):
            shift_idx = x_decoded[idx]
            shifts_covered[shift_idx] += 1
            total_cost += self.data["shift_costs"][shift_idx]
        coverage_diff = self.data["shift_requirements"] - shifts_covered
        coverage_penalty = np.sum(np.abs(coverage_diff))
        return total_cost + coverage_penalty


problem = EmployeeRosteringProblem(
    variables=[DiscreteVariable(choices=[*range(0, num_shifts)], name=f"shift_var_{i}") for i in range(num_employees)],
    minmax="min",
    data=data
)

config = FireflySwarmOptimizationConfig(
    population_size=20,
    fitness_error=0.1,
    max_cycles=100,
    alpha=0.5,
    beta_min=0.2,
    gamma=0.99,
)
result = FireflySwarmOptimization(config).optimize(problem)


print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {problem.transform_solution(result.best_solution.position)}")  # Decoded (Real) solution
