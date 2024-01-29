# Define a chromosome representation that encodes the allocation of resources and patient flow in the emergency
# department. This could be a binary matrix where each row represents a patient and each column represents a resource
# (room). If the element is 1, it means the patient is assigned to that particular room, and if the element is 0,
# it means the patient is not assigned to that room
#
# Please note that this implementation is a basic template and may require further customization based on the
# specific objectives, constraints, and evaluation criteria of your healthcare workflow optimization problem. You'll
# need to define the specific fitness function and optimization objectives based on the factors you want to optimize,
# such as patient waiting times, resource utilization, and other relevant metrics in the healthcare workflow context.

import numpy as np
from pyvolutionary import BinaryVariable, Task, HarmonySearchOptimization, HarmonySearchOptimizationConfig

# Define the problem parameters
num_patients = 50  # Number of patients
num_resources = 10  # Number of resources (room)

# Define the patient waiting time matrix (randomly generated for the example)
# Why? May be, doctors need time to prepare tools,...
waiting_matrix = np.random.randint(1, 10, size=(num_patients, num_resources))

data = {
    "num_patients": num_patients,
    "num_resources": num_resources,
    "waiting_matrix": waiting_matrix,
    "max_resource_capacity": 10,  # Maximum capacity of each room
    "max_waiting_time": 60,  # Maximum waiting time
    "penalty_value": 1e2,  # Define a penalty value
    "penalty_patient": 1e10
}


class SupplyChainProblem(Task):
    def objective_function(self, x):
        x_transformed = np.array(
            self.transform_solution(x)["placement_var"]
        ).reshape(self.data["num_patients"], self.data["num_resources"])

        # If any row has all 0 value, it indicates that this patient is not allocated to any room.
        # If a patient is assigned to more than 3 room, not allow
        if np.any(np.all(x_transformed == 0, axis=1)) or np.any(np.sum(x_transformed > 3, axis=1)):
            return self.data["penalty_patient"]

        # Calculate fitness based on optimization objectives
        room_used = np.sum(x_transformed, axis=0)
        wait_time = np.sum(x_transformed * self.data["waiting_matrix"], axis=1)
        violated_constraints = np.sum(room_used > self.data["max_resource_capacity"]) + np.sum(
            wait_time > self.data["max_waiting_time"]
        )

        # Calculate the fitness value based on the objectives
        resource_utilization_fitness = 1 - np.mean(room_used) / self.data["max_resource_capacity"]
        waiting_time_fitness = 1 - np.mean(wait_time) / self.data["max_waiting_time"]

        fitness = resource_utilization_fitness + waiting_time_fitness + (
            self.data["penalty_value"] * violated_constraints
        )
        return fitness


problem = SupplyChainProblem(
    variables=[BinaryVariable(n_vars=num_patients * num_resources, name="placement_var")],
    minmax="min",
    data=data,
)

config = HarmonySearchOptimizationConfig(
    population_size=20,
    fitness_error=0.1,
    max_cycles=50,
    consideration_rate=0.15,
    pitch_adjusting_rate=0.5,
)
result = HarmonySearchOptimization(config).optimize(problem)

best_scheduling = np.array(problem.transform_solution(result.best_solution.position)["placement_var"]).reshape(
    (num_patients, num_resources)
)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {best_scheduling}")  # Decoded (Real) solution
