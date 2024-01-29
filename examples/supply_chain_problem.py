# Let's assume we have a supply chain network with 5 distribution centers (DC1, DC2, DC3, DC4, DC5) and 10 products (
# P1, P2, P3, ..., P10). Our goal is to determine the optimal allocation of products to the distribution centers in a
# way that minimizes the total transportation cost.
#
# Each solution represents an allocation of products to distribution centers. We can use a binary matrix with
# dimensions (10, 5) where each element (i, j) represents whether product i is allocated to distribution center j.
# For example, a chromosome [1, 0, 1, 0, 1] would mean that product 1 is allocated to DC1, DC3, DC5.
#
# We can add the maximum capacity of each distribution center, therefor we need penalty term to the fitness
# evaluation function to penalize solutions that violate this constraint. The penalty can be based on the degree of
# violation or a fixed penalty value.

from typing import Any
import numpy as np
from pyvolutionary import BinaryVariable, Task, WhalesOptimization, WhalesOptimizationConfig

# Define the problem parameters
num_products = 10
num_distribution_centers = 5

# Define the transportation cost matrix (randomly generated for the example)
transportation_cost = np.random.randint(1, 10, size=(num_products, num_distribution_centers))

data = {
    "num_products": num_products,
    "num_distribution_centers": num_distribution_centers,
    "transportation_cost": transportation_cost,
    "max_capacity": 4,  # Maximum capacity of each distribution center
    "penalty": 1e10  # Define a penalty value for maximum capacity of each distribution center
}


class SupplyChainProblem(Task):
    def objective_function(self, x: list[Any]):
        x_decoded = np.array(self.transform_solution(x)["placement_var"])
        x_decoded = x_decoded.reshape((self.data["num_products"], self.data["num_distribution_centers"]))

        if np.any(np.all(x_decoded == 0, axis=1)):
            # If any row has all 0 value, it indicates that this product is not allocated to any distribution center.
            return 0

        total_cost = np.sum(self.data["transportation_cost"] * x_decoded)
        # Penalty for violating maximum capacity constraint
        excess_capacity = np.maximum(np.sum(x_decoded, axis=0) - self.data["max_capacity"], 0)
        penalty = np.sum(excess_capacity)
        # Calculate fitness value as the inverse of the total cost plus the penalty
        fitness = 1 / (total_cost + penalty)
        return fitness


problem = SupplyChainProblem(
    variables=[BinaryVariable(n_vars=num_products * num_distribution_centers, name="placement_var")],
    minmax="max",
    data=data,
)

config = WhalesOptimizationConfig(population_size=20, fitness_error=0.1, max_cycles=50)
result = WhalesOptimization(config).optimize(problem)

best_scheduling = np.array(problem.transform_solution(result.best_solution.position)["placement_var"]).reshape(
    (num_products, num_distribution_centers)
)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {best_scheduling}")  # Decoded (Real) solution
