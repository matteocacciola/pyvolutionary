import pytest

from pyvolutionary import (
    ContinuousVariable,
    OptimizationResult,
    AfricanVultureOptimization,
    AfricanVultureOptimizationConfig,
    Task,
)




from typing import Any
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics

from pyvolutionary import (
    best_agent,
    ContinuousVariable,
    DiscreteVariable,
    Task,
    TaskType,
    ZebraOptimization,
    ZebraOptimizationConfig,
)

# Load the data set; In this example, the breast cancer dataset is loaded.
X, y = datasets.load_breast_cancer(return_X_y=True)

# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


class SvmOptimizedProblem(Task):
    def objective_function(self, x: list[Any]):
        x_transformed = self.transform_position(x)
        C, kernel = x_transformed["C"], x_transformed["kernel"]
        degree, gamma = x_transformed["degree"], x_transformed["gamma"]

        svc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True, random_state=1)
        svc.fit(X_train_std, y_train)
        y_predict = svc.predict(X_test_std)
        return metrics.accuracy_score(y_test, y_predict)


task = SvmOptimizedProblem(
    variables=[
        ContinuousVariable(lower_bound=0.01, upper_bound=1000., name="C"),
        DiscreteVariable(choices=["linear", "poly", "rbf", "sigmoid"], name="kernel"),
        DiscreteVariable(choices=[*range(1, 6)], name="degree"),
        DiscreteVariable(choices=['scale', 'auto', 0.01, 0.05, 0.1, 0.5, 1.0], name="gamma"),
    ],
    minmax=TaskType.MAX,
)

# configuration = ParticleSwarmOptimizationConfig(
#     population_size=200,
#     fitness_error=10e-4,
#     max_cycles=100,
#     c1=0.1,
#     c2=0.1,
#     w=[0.35, 1],
# )
configuration = ZebraOptimizationConfig(
    population_size=20,
    fitness_error=10e-4,
    max_cycles=100,
)

result = ZebraOptimization(configuration, True).optimize(task)
best = best_agent(result.evolution[-1].agents, task.minmax)

print(f"Best parameters: {task.transform_position(best.position)}")
print(f"Best accuracy: {best.cost}")








class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        return -sum(xi ** 2 for xi in x)


@pytest.fixture
def data() -> tuple[AfricanVultureOptimizationConfig, Task]:
    config = AfricanVultureOptimizationConfig(
        population_size=20,
        fitness_error=0.01,
        max_cycles=100,
        p=[0.6, 0.4, 0.6],
        alpha=0.8,
        gamma=2.5,
    )
    dimension = 2
    task = Sphere(
        variables=[ContinuousVariable(name=f"x{i}", lower_bound=-10, upper_bound=10) for i in range(dimension)],
        minmax="max",
    )

    return config, task


def test_valid_optimization(data):
    optimization_config, task = data

    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task)
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_threads(data):
    optimization_config, task = data

    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="thread")
    assert isinstance(result, OptimizationResult)


def test_valid_optimization_with_processes(data):
    optimization_config, task = data

    o = AfricanVultureOptimization(optimization_config)
    result = o.optimize(task, mode="process")
    assert isinstance(result, OptimizationResult)