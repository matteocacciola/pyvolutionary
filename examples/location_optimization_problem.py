# Let's consider an example of location optimization in the context of a retail company that wants to open a certain
# number of new stores in a region to maximize market coverage while minimizing operational costs.
#
# A company wants to open five new stores in a region with several potential locations. The objective is to determine
# the optimal locations for these stores while considering factors such as population density and transportation
# costs. The goal is to maximize market coverage by locating stores in areas with high demand while minimizing the
# overall transportation costs required to serve customers.
#
# By applying location optimization techniques, the retail company can make informed decisions about where to open
# new stores, considering factors such as population density and transportation costs. This approach allows the
# company to maximize market coverage, make efficient use of resources, and ultimately improve customer service and
# profitability.
#
# Note that this example is a simplified illustration, and in real-world scenarios, location optimization problems
# can involve more complex constraints, additional factors, and larger datasets. However, the general process remains
# similar, involving data analysis, mathematical modeling, and optimization techniques to determine the optimal
# locations for facilities.

import numpy as np
from pyvolutionary import BinaryVariable, Task, MothFlameOptimization, MothFlameOptimizationConfig

# Define the coordinates of potential store locations
locations = np.array([
    [2, 4],
    [5, 6],
    [9, 3],
    [7, 8],
    [1, 10],
    [3, 2],
    [5, 5],
    [8, 2],
    [7, 6],
    [1, 9]
])
# Define the transportation costs matrix based on the Euclidean distance between locations
distance_matrix = np.linalg.norm(locations[:, np.newaxis] - locations, axis=2)

# Define the number of stores to open
num_stores = 5

# Define the maximum distance a customer should travel to reach a store
max_distance = 10

data = {
    "num_stores": num_stores,
    "max_distance": max_distance,
    "penalty": 1e10
}


class LocationOptProblem(Task):
    def objective_function(self, x):
        x_transformed = np.array(self.transform_solution(x)["placement_var"])
        total_coverage = np.sum(x_transformed)
        total_dist = np.sum(x_transformed[:, np.newaxis] * distance_matrix)
        if total_dist == 0 or total_coverage < self.data["num_stores"]:  # Penalize solutions with fewer stores
            return self._EPS
        return total_dist


problem = LocationOptProblem(
    variables=[BinaryVariable(n_vars=len(locations), name="placement_var")],
    minmax="min",
    data=data,
)

config = MothFlameOptimizationConfig(population_size=20, fitness_error=0.1, max_cycles=100)
result = MothFlameOptimization(config, debug=True).optimize(problem)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {problem.transform_solution(result.best_solution.position)}")  # Decoded (Real) solution
