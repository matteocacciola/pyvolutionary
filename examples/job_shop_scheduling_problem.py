import numpy as np
from pyvolutionary import PermutationVariable, Task, GeneticAlgorithmOptimization, GeneticAlgorithmOptimizationConfig

job_times = [[2, 1, 3], [4, 2, 1], [3, 3, 2]]
machine_times = [[3, 2, 1], [1, 4, 2], [2, 3, 2]]

n_jobs = len(job_times)
n_machines = len(machine_times)

data = {
    "job_times": job_times,
    "machine_times": machine_times,
    "n_jobs": n_jobs,
    "n_machines": n_machines
}


class JobShopProblem(Task):
    def objective_function(self, x):
        x_transformed = self.transform_solution(x)
        makespan = np.zeros((self.data["n_jobs"], self.data["n_machines"])).tolist()
        for gene in x_transformed["per_var"]:
            job_idx = gene // self.data["n_machines"]
            machine_idx = gene % self.data["n_machines"]
            if job_idx == 0 and machine_idx == 0:
                makespan[job_idx][machine_idx] = job_times[job_idx][machine_idx]
            elif job_idx == 0:
                makespan[job_idx][machine_idx] = makespan[job_idx][machine_idx - 1] + job_times[job_idx][machine_idx]
            elif machine_idx == 0:
                makespan[job_idx][machine_idx] = makespan[job_idx - 1][machine_idx] + job_times[job_idx][machine_idx]
            else:
                makespan[job_idx][machine_idx] = max(
                    makespan[job_idx][machine_idx - 1], makespan[job_idx - 1][machine_idx]
                ) + job_times[job_idx][machine_idx]
        return np.max(makespan)


problem = JobShopProblem(
    variables=[PermutationVariable(items=list(range(0, n_jobs * n_machines)), name="per_var")],
    minmax="min",
    data=data,
)

config = GeneticAlgorithmOptimizationConfig(
    population_size=20,
    fitness_error=0.1,
    max_cycles=100,
    px_over=0.8,
)
result = GeneticAlgorithmOptimization(config).optimize(problem)

print(f"Best agent: {result.best_solution}")  # Encoded solution
print(f"Best solution: {result.best_solution.position}")  # Encoded solution
print(f"Best fitness: {result.best_solution.cost}")
print(f"Best real scheduling: {problem.transform_solution(result.best_solution.position)}")  # Decoded (Real) solution
