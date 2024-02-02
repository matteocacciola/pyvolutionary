import numpy as np
from pyvolutionary import PermutationVariable, Task, ArchimedeOptimization, ArchimedeOptimizationConfig

# Define the graph representation
data = {
    "graph": np.array([
        [0, 2, 4, 0, 7, 9],
        [2, 0, 1, 4, 2, 8],
        [4, 1, 0, 1, 3, 0],
        [6, 4, 5, 0, 3, 2],
        [0, 2, 3, 3, 0, 2],
        [9, 0, 4, 2, 2, 0]
    ])
}


class ShortestPathProblem(Task):
    def objective_function(self, x):
        individual = self.transform_solution(x)["path"]
        total_distance = 0
        for idx in range(len(individual) - 1):
            start_node = individual[idx]
            end_node = individual[idx + 1]
            weight = self.data["graph"][start_node, end_node]
            if weight == 0:
                return self._EPS
            total_distance += weight
        return total_distance


problem = ShortestPathProblem(
    variables=[PermutationVariable(items=list(range(0, len(data["graph"]))), name="path")],
    minmax="min",
    data=data,
)

config = ArchimedeOptimizationConfig(
    population_size=20,
    fitness_error=0.1,
    max_cycles=100,
    c1=2.0,
    c2=2.0,
    c3=2.0,
    c4=0.5,
    acc=[0.2, 0.9],
)
result = ArchimedeOptimization(config, debug=True).optimize(problem)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {problem.transform_solution(result.best_solution.position)}")  # Decoded (Real) solution
