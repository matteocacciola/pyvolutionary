# pyVolutionary

[![GitHub release](https://img.shields.io/badge/release-1.1.0-yellow.svg)](https://github.com/matteocacciola/pyvolutionary/releases/tag/v1.1.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvolutionary.svg)
![PyPI - Status](https://img.shields.io/pypi/status/pyvolutionary.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyvolutionary.svg)
![GitHub Release Date](https://img.shields.io/github/release-date/matteocacciola/pyvolutionary.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Introduction
<img style="width: 200px; background-color: #fff; float: left; margin: 0 15px 0 0;" src="logo.png" alt="pyVolutionary"/>

**pyVolutionary** stands as a versatile Python library dedicated to metaheuristic algorithms within the realm of
evolutionary computation. Engineered for ease of use, flexibility, and speed, it exhibits robustness and efficiency,
having undergone rigorous testing on large-scale problem instances. The primary objectives encompass the implementation
of both classical and cutting-edge nature-inspired algorithms. The library is conceived as a user-friendly resource
facilitating rapid access to optimization algorithms for researchers, fostering the dissemination of optimization
knowledge to a broad audience without financial barriers.

Nature-inspired algorithms constitute a widely embraced tool for addressing optimization challenges. Over the course of
their evolution, a plethora of variants have emerged ([paper 1](https://arxiv.org/abs/1307.4186),
[paper 2](https://www.mdpi.com/2076-3417/8/9/1521)), showcasing their adaptability and versatility across diverse domains
and applications. Noteworthy advancements have been achieved through hybridization, modification, and adaptation of these
algorithms. However, the implementation of nature-inspired algorithms can often pose a formidable challenge, characterized
by complexity and tedium. **pyVolutionary** is specifically crafted to surmount this challenge, offering a streamlined and
expedited approach to leveraging these algorithms without the need for arduous, time-consuming implementations from scratch.

The list of algorithms currently implemented in **pyVolutionary** can be consulted in the [Algorithms](#Algorithms) section.
The library is continuously updated with new algorithms and problems, and contributions are welcome.

## Installation
**pyVolutionary** is available on [PyPI](https://pypi.org/project/pyvolutionary/), and can be installed via pip:

```bash
pip install pyvolutionary
```

## Usage
Once installed, **pyVolutionary** can be imported into your Python scripts as follows:

```python
import pyvolutionary
```

Now, you can access the algorithms and problems included in the library. With **pyVolutionary**, you can solve both
continuous and discrete optimization problems. It is also possible to solve mixed problems, i.e., problems with both
continuous and discrete variables. In order to do so, you need to define a `Task` class, which inherits from the
`Task` class of the library. The list of variables in the problem must be specified in the constructor of the class
inheriting from `Task`. The variables can be either continuous or discrete. The following table describes the types of
variables currently implemented in the library.

| Variable type | Class name            | Description                            | Example                                                                          |
|---------------|-----------------------|----------------------------------------|----------------------------------------------------------------------------------|
| Continuous    | `ContinuousVariable`  | A continuous variable                  | `ContinuousVariable(name="x0", lower_bound=-100.0, upper_bound=100.0)`           |
| Discrete      | `DiscreteVariable`    | A discrete variable                    | `DiscreteVariable(choices=["scale", "auto", 0.01, 0.1, 0.5, 1.0], name="gamma")` |
| Permutation   | `PermutationVariable` | A permutation of the specified choices | `PermutationVariable(items=[[60, 200], [180, 200], [80, 180]], name="routes")`   |

An example of a custom `Task` class is the following:

```python
from pyvolutionary import ContinuousVariable, Task

class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        x1, x2 = x
        f1 = x1 - 2 * x2 + 3
        f2 = 2 * x1 + x2 - 8
        return f1 ** 2 + f2 ** 2


# Define the task with the bounds and the configuration of the optimizer
task = Sphere(
    variables=[
        ContinuousVariable(name="x1", lower_bound=-100.0, upper_bound=100.0),
        ContinuousVariable(name="x2", lower_bound=-100.0, upper_bound=100.0),
    ],
)
```

You can pass the `minmax` parameter to the `Task` class to specify whether you want to minimize or maximize the function.
Therefore, if you want to maximize the function, you can write:

```python
from pyvolutionary import ContinuousVariable, Task

class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        x1, x2 = x
        f1 = x1 - 2 * x2 + 3
        f2 = 2 * x1 + x2 - 8
        return -(f1 ** 2 + f2 ** 2)

task = Sphere(
    variables=[
        ContinuousVariable(name="x1", lower_bound=-100.0, upper_bound=100.0),
        ContinuousVariable(name="x2", lower_bound=-100.0, upper_bound=100.0),
    ],
    minmax="max",
)
```

By default, the `minmax` parameter is set to `min`. If necessary (e.g., in the implementation of the objective function),
additional data can be injected into the `Task` class by using the `data` parameter of the constructor. This data can
be accessed by using the `data` attribute of the `Task` class (see combinatorial example below).

Finally, you can also specify the seed of the random number generator by using the `seed` parameter of the definition
of the `Task`:

```python
from pyvolutionary import ContinuousVariable, Task

class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        x1, x2 = x
        f1 = x1 - 2 * x2 + 3
        f2 = 2 * x1 + x2 - 8
        return -(f1 ** 2 + f2 ** 2)

task = Sphere(
    variables=[
        ContinuousVariable(name="x1", lower_bound=-100.0, upper_bound=100.0),
        ContinuousVariable(name="x2", lower_bound=-100.0, upper_bound=100.0),
    ],
    minmax="max",
    seed=42,
)
```

### Continuous problems
For example, let us inspect how you can solve the continuous _sphere_ problem with the Particle Swarm Optimization algorithm.

```python
from pyvolutionary import ContinuousVariable, ParticleSwarmOptimization, ParticleSwarmOptimizationConfig, Task

# Define the problem, you can replace the following class with your custom problem to optimize
class Sphere(Task):
    def objective_function(self, x: list[float]) -> float:
        x1, x2 = x
        f1 = x1 - 2 * x2 + 3
        f2 = 2 * x1 + x2 - 8
        return f1 ** 2 + f2 ** 2


# Define the task with the bounds and the configuration of the optimizer
population = 200
dimension = 2
position_min = -100.0
position_max = 100.0
generation = 400
fitness_error = 10e-4
task = Sphere(
    variables=[ContinuousVariable(
        name=f"x{i}", lower_bound=position_min, upper_bound=position_max
    ) for i in range(dimension)],
)

configuration = ParticleSwarmOptimizationConfig(
    population_size=population,
    fitness_error=fitness_error,
    max_cycles=generation,
    c1=0.1,
    c2=0.1,
    w=[0.35, 1],
)
optimization_result = ParticleSwarmOptimization(configuration).optimize(task)
```

You can also specify the mode of the solver by using the `mode` argument of the `optimize` method.
For instance, if you want to run the Particle Swarm Optimization algorithm in parallel with threads, you can write:

```python
optimization_result = ParticleSwarmOptimization(configuration).optimize(task, mode="thread")
```

The possible values of the `mode` parameter are:
- `serial`: the algorithm is run in serial mode;
- `process`: the algorithm is run in parallel with processes;
- `thread`: the algorithm is run in parallel with threads.

In case of `process` and `thread` modes, you can also specify the number of processes or threads to use by using the
`n_jobs` argument of the `optimize` method:

```python
optimization_result = ParticleSwarmOptimization(configuration).optimize(task, mode="thread", jobs=4)
```

The optimization result is a dictionary containing the following keys:
- `evolution`: a list of the agents found at each generation
- `rates`: a list of the fitness values of the agents found at each generation
- `best_solution`: the best agent found by the algorithm

Explicitly, the `evolution` key contains a list of `Population`, i.e. a dictionary which `agents` key contains a list of
`Agent`. The latter is a dictionary composed by the following basic keys:
- `position`: the position of the agent
- `fitness`: the fitness value of the agent
- `cost`: the cost of the agent

```python
from pydantic import BaseModel

class Agent(BaseModel):
    position: list[float]
    cost: float
    fitness: float
```

These are the basic information, but each algorithm can add more information to the agent, such as the velocity in the
case of PSO.

### Discrete problem
A typical problem involving discrete variables is the optimization of the hyperparameters of a Machine Learning model,
such as a Support Vector Classifier (SVC). You can use **pyVolutionary** to accomplish this task. In the following, we
provide an example using the Particle Swarm Optimization (PSO) as the optimizer.

```python
from typing import Any
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics

from pyvolutionary import (
    best_agent,
    ContinuousVariable,
    DiscreteVariable,
    ParticleSwarmOptimization,
    ParticleSwarmOptimizationConfig,
    Task,
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
        DiscreteVariable(choices=["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1.0], name="gamma"),
    ],
    minmax="max",
)

configuration = ParticleSwarmOptimizationConfig(
    population_size=200,
    fitness_error=10e-4,
    max_cycles=100,
    c1=0.1,
    c2=0.1,
    w=[0.35, 1],
)

result = ParticleSwarmOptimization(configuration).optimize(task)
best = best_agent(result.evolution[-1].agents, task.minmax)

print(f"Best parameters: {task.transform_position(best.position)}")
print(f"Best accuracy: {best.cost}")
```

You can replace the PSO with any other algorithm implemented in the library.

### Combinatorial problem
Within the framework of **pyVolutionary** for addressing the Traveling Salesman Problem (TSP), a solution is a plausible
route signifying a tour that encompasses visiting all cities precisely once and returning to the initial city.
Typically, this solution is articulated as a permutation of the cities, wherein each city features exactly once in the permutation.

As an illustration, consider a TSP scenario involving 5 cities denoted as A, B, C, D, and E. A potential solution might
be denoted by the permutation [A, B, D, E, C], illustrating the order in which the cities are visited. This interpretation
indicates that the tour initiates at city A, proceeds to city B, then D, E, and ultimately C before looping back to city A.

The following code snippet illustrates how to solve the TSP with the Virus Colony Search Optimization algorithm.

```python
from typing import Any
import numpy as np
from pyvolutionary import (
    best_agent,
    Task,
    PermutationVariable,
    VirusColonySearchOptimization,
    VirusColonySearchOptimizationConfig,
)
from pyvolutionary.helpers import distance


class TspProblem(Task):
    def objective_function(self, x: list[Any]) -> float:
        x_transformed = self.transform_position(x)
        routes = x_transformed["routes"]
        city_pos = self.data["city_positions"]
        n_routes = len(routes)
        return np.sum([distance(
            city_pos[route], city_pos[routes[(idx + 1) % n_routes]]
        ) for idx, route in enumerate(routes)])


city_positions = [
    [60, 200], [180, 200], [80, 180], [140, 180], [20, 160],
    [100, 160], [200, 160], [140, 140], [40, 120], [100, 120],
    [180, 100], [60, 80], [120, 80], [180, 60], [20, 40],
    [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]
]
task = TspProblem(
    variables=[PermutationVariable(name="routes", items=list(range(0, len(city_positions))))],
    data={"city_positions": city_positions},
)
configuration = VirusColonySearchOptimizationConfig(
    population_size=10,
    fitness_error=0.01,
    max_cycles=100,
    lamda=0.1,
    sigma=2.5,
)
result = VirusColonySearchOptimization(configuration).optimize(task)
best = best_agent(result.evolution[-1].agents, task.minmax)

print(f"Best real scheduling: {task.transform_position(best.position)}")
print(f"Best fitness: {best.cost}")
```

### Utilities
**pyVolutionary** provides a set of utilities to facilitate the use of the library. For example, you can use the
`plot` function to plot the evolution of the algorithm. Its usage is as follows:

```python
plot(function: callable, pos_min: float, pos_max: float, evolution: list[Population])
```
where:
- `function`: the function to plot, i.e., the function to optimize
- `pos_min`: the minimum possible coordinates in the search space
- `pos_max`: the maximum possible coordinates in the search space
- `evolution`: the evolution of the algorithm, i.e., the list of the agents found at each generation

It is also possible to inspect an animation of the evolution of the algorithm by using the `animate` function:

```python
animate(function: callable, optimization_result: OptimizationResult, pos_min: float, pos_max: float, filename: str)
```

where:
- `function`: the same as above
- `optimization_result`: the result of the optimization, i.e., the dictionary returned by the `optimize` method
- `pos_min`: the same as above
- `pos_max`: the same as above
- `filename`: the name of the file where to save the animation

Furthermore, you can extract the trend of the best agent found by the algorithm by using the `best_agent_trend` function:

```python
best_agent_trend(optimization_result: OptimizationResult, iters: list[int] | None = None) -> list[float]
```

where:
- `optimization_result`: the same as above
- `iters`: a list of the iterations to consider. If `None`, all the iterations are considered
It returns a list of the cost values of the best agent found at each iteration.

If you prefer, you can extract the trend of a specific agent by using the `agent_trend` function:

```python
agent_trend(optimization_result: OptimizationResult, idx: int, iters: list[int] | None = None) -> list[float]
```

where:
- `optimization_result`: the same as above
- `idx`: the index of the agent to consider
- `iters`: the same as above
It returns a list of the cost values of the agent at each iteration.

## Extending the library
**pyVolutionary** is designed to be easily extensible. You can add your own algorithms and problems to the library by
following the instructions below.

### Adding a new algorithm
To add a new algorithm, you need to create a new class that inherits from the `OptimizationAbstract` class. The new
class must implement the `optimization_step` method, where you can implement your new metaheuristic algorithm.

The constructor of the new class must accept a `config` parameter, which is a Pydantic model extending the `BaseOptimizationConfig`
class. This class contains the parameters of the algorithm, such as the population size, the number of generations, etc.

```python
from pydantic import BaseModel

class BaseOptimizationConfig(BaseModel):
    population_size: int
    fitness_error: float | None = None
    max_cycles: int
```

The examples listed in the following section can be used as a reference for the implementation of a new algorithm.

Once you created your new classes, you can run the algorithm by calling the `optimize` method, which takes as input a
`Task` object and returns a dictionary as above described.

## Algorithms
The following algorithms are currently implemented in **pyVolutionary**:

| Algorithm                                 | Class                                    | Year | Paper                                                                                                                                                                                                                         | Example                                                                               |
|-------------------------------------------|------------------------------------------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| African Vulture Optimization              | `AfricanVultureOptimization`             | 2022 | [paper](https://doi.org/10.1016/j.cie.2021.107408)                                                                                                                                                                            | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/avo.py)   |
| Ant Colony Optimization                   | `AntColonyOptimization`                  | 2008 | [paper](https://doi.org/10.1109/MCI.2006.329691)                                                                                                                                                                              | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/aco.py)   |
| Aquila Optimization                       | `AquilaOptimization`                     | 2021 | [paper](https://doi.org/10.1016/j.cie.2021.107250)                                                                                                                                                                            | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/ao.py)    |
| Artificial Bee Colony Optimization        | `BeeColonyOptimization`                  | 2007 | [paper](https://api.semanticscholar.org/CorpusID:8215393)                                                                                                                                                                     | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/bco.py)   |
| Bacterial Foraging Optimization           | `BacterialForagingOptimization`          | 2002 | [paper](https://api.semanticscholar.org/CorpusID:108291966)                                                                                                                                                                   | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/bfo.py)   |
| Bat Optimization                          | `BatOptimization`                        | 2010 | [paper](https://link.springer.com/chapter/10.1007/978-3-642-12538-6_6)                                                                                                                                                        | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/bo.py)    |
| Biogeography-Based Optimization           | `BiogeographyBasedOptimization`          | 2008 | [paper](https://ieeexplore.ieee.org/abstract/document/4475427)                                                                                                                                                                | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/bgbo.py)  |
| Brown-Bear Optimization                   | `BrownBearOptimization`                  | 2023 | [paper](https://www.taylorfrancis.com/chapters/edit/10.1201/9781003337003-6/novel-brown-bear-optimization-algorithm-solving-economic-dispatch-problem-tapan-prakash-praveen-prakash-singh-vinay-pratap-singh-sri-niwas-singh) | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/bbo.py)   |
| Camel Caravan Optimization                | `CamelCaravanOptimization`               | 2016 | [paper](https://ijeee.edu.iq/Papers/Vol12-Issue2/118375.pdf)                                                                                                                                                                  | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/ccoa.py)  |
| Cat Swarm Optimization                    | `CatSwarmOptimization`                   | 2006 | [paper](https://link.springer.com/chapter/10.1007/978-3-540-36668-3_94)                                                                                                                                                       | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/cso.py)   |
| Chernobyl Disaster Optimization           | `ChernobylDisasterOptimization`          | 2023 | [paper](https://link.springer.com/article/10.1007/s00521-023-08261-1)                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/cdo.py)   |
| Coral Reef Optimization                   | `CoralReefOptimization`                  | 2014 | [paper](https://www.hindawi.com/journals/tswj/2014/739768/)                                                                                                                                                                   | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/cro.py)   |
| Cuckoo Search Optimization                | `CuckooSearchOptimization`               | 2009 | [paper](https://doi.org/10.1109/NABIC.2009.5393690)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/csoa.py)  |
| Coyotes Optimization                      | `CoyotesOptimization`                    | 2018 | [paper](https://ieeexplore.ieee.org/document/8477769)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/co.py)    |
| Earthworms Optimization                   | `EarthwormsOptimization`                 | 2015 | [paper](https://www.inderscience.com/info/inarticle.php?artid=93328)                                                                                                                                                          | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/ewo.py)   |
| Electromagnetic Field Optimization        | `ElectromagneticFieldOptimization`       | 2016 | [paper](https://doi.org/10.1016/j.swevo.2015.07.002)                                                                                                                                                                          | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/efo.py)   |
| Elephant Herd Optimization                | `ElephantHerdOptimization`               | 2015 | [paper](https://ieeexplore.ieee.org/document/7383528)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/eho.py)   |
| Energy Valley Optimization                | `EnergyValleyOptimization`               | 2023 | [paper](https://www.nature.com/articles/s41598-022-27344-y)                                                                                                                                                                   | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/evo.py)   |
| Firefly Swarm Optimization                | `FireflySwarmOptimization`               | 2009 | [paper](https://doi.org/10.1504/IJBIC.2010.032124)                                                                                                                                                                            | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/fso.py)   |
| Fire Hawk Optimization                    | `FireHawkOptimization`                   | 2022 | [paper](https://link.springer.com/article/10.1007/s10462-022-10173-w)                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/fho.py)   |
| Fireworks Optimization                    | `FireworksOptimization`                  | 2010 | [paper](https://doi.org/10.1504/IJBIC.2010.037285)                                                                                                                                                                            | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/fwa.py)   |
| Fish School Search Optimization           | `FishSchoolSearchOptimization`           | 2008 | [paper](https://doi.org/10.1109/ICSMC.2008.4811695)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/fsso.py)  |
| Flower Pollination Algorithm Optimization | `FlowerPollinationAlgorithmOptimization` | 2012 | [paper](https://link.springer.com/chapter/10.1007/978-3-642-32894-7_27)                                                                                                                                                       | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/fpao.py)  |
| Forest Optimization Algorithm             | `ForestOptimizationAlgorithm`            | 2014 | [paper](https://doi.org/10.1016/j.eswa.2014.05.009)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/foa.py)   |
| Fox Optimization                          | `FoxOptimization`                        | 2023 | [paper](https://link.springer.com/article/10.1007/s10489-022-03533-0)                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/fo.py)    |
| Genetic Algorithm Optimization            | `GeneticAlgorithmOptimization`           | 1989 | [paper](https://www.sciencedirect.com/book/9780128219867/nature-inspired-optimization-algorithms)                                                                                                                             | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/gao.py)   |
| Giza Pyramid Construction Optimization    | `GizaPyramidConstructionOptimization`    | 2021 | [paper](https://doi.org/10.1007/s12065-020-00451-3)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/gpco.py)  |
| Grasshopper Optimization Algorithm        | `GrasshopperOptimization`                | 2017 | [paper](https://doi.org/10.1016/j.advengsoft.2017.01.00)                                                                                                                                                                      | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/goa.py)   |
| Grey Wolf Optimization                    | `GreyWolfOptimization`                   | 2014 | [paper](https://doi.org/10.1016/j.advengsoft.2013.12.007)                                                                                                                                                                     | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/gwo.py)   |
| Harmony Search Optimization               | `HarmonySearchOptimization`              | 2001 | [paper](https://doi.org/10.1177/003754970107600201)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/hso.py)   |
| Imperialist Competitive Optimization      | `ImperialistCompetitiveOptimization`     | 2013 | [paper](https://doi.org/10.1109/ICCKE.2013.6687751)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/ico.py)   |
| Invasive Weed Optimization                | `InvasiveWeedOptimization`               | 2006 | [paper](https://doi.org/10.1016/j.ecoinf.2006.07.003)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/iwo.py)   |
| Krill Herd Optimization                   | `KrillHerdOptimization`                  | 2012 | [paper](https://doi.org/10.1016/j.cnsns.2012.05.010)                                                                                                                                                                          | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/kho.py)   |
| Levy Flight Jaya Swarm Optimization       | `LeviFlightJayaSwarmOptimization`        | 2021 | [paper](https://doi.org/10.1016/j.eswa.2020.113902)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/lfjso.py) |
| Monarch Butterfly Optimization            | `MonarchButterflyOptimization`           | 2019 | [paper](https://link.springer.com/article/10.1007/s00521-015-1923-y)                                                                                                                                                          | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/mbo.py)   |
| Mountain Gazelle Optimization             | `MountainGazelleOptimization`            | 2022 | [paper](https://doi.org/10.1016/j.advengsoft.2022.103282)                                                                                                                                                                     | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/mgo.py)   |
| Osprey Optimization                       | `OspreyOptimization`                     | 2023 | [paper](https://www.frontiersin.org/articles/10.3389/fmech.2022.1126450/full)                                                                                                                                                 | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/ooa.py)   |
| Particle Swarm Optimization               | `ParticleSwarmOptimization`              | 1995 | [paper](https://ieeexplore.ieee.org/document/488968)                                                                                                                                                                          | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/pso.py)   |
| Pathfinder Algorithm Optimization         | `PathfinderAlgorithmOptimization`        | 2019 | [paper](https://doi.org/10.1016/j.asoc.2019.03.012)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/pfao.py)  |
| Pelican Optimization                      | `PelicanOptimization`                    | 2022 | [paper](https://www.mdpi.com/1424-8220/22/3/855)                                                                                                                                                                              | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/poa.py)   |
| Seagull Optimization                      | `SeagullOptimization`                    | 2019 | [paper](https://doi.org/10.1016/j.knosys.2018.11.024)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/so.py)    |
| Siberian Tiger Optimization               | `SiberianTigerOptimization`              | 2022 | [paper](https://ieeexplore.ieee.org/document/9989374)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/stoa.py)  |
| Tasmanian Devil Optimization              | `TasmanianDevilOptimization`             | 2022 | [paper](https://ieeexplore.ieee.org/document/9714388)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/tdo.py)   |
| Virus Colony Search Optimization          | `VirusColonySearchOptimization`          | 2016 | [paper](https://doi.org/10.1016/j.advengsoft.2015.11.004)                                                                                                                                                                     | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/vcso.py)  |
| Walrus Optimization                       | `WalrusOptimization`                     | 2022 | [paper](http://doi.org/10.21203/rs.3.rs-2174098/v1)                                                                                                                                                                           | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/woa.py)   |
| Whales Optimization                       | `WhalesOptimization`                     | 2016 | [paper](https://doi.org/10.1016/j.advengsoft.2016.01.008)                                                                                                                                                                     | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/wo.py)    |
| Wildebeest Herd Optimization              | `WildebeestHerdOptimization`             | 2019 | [paper](https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs190495)                                                                                                                             | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/who.py)   |
| Zebra Optimization                        | `ZebraOptimization`                      | 2022 | [paper](https://ieeexplore.ieee.org/document/9768820)                                                                                                                                                                         | [example](https://github.com/matteocacciola/pyvolutionary/tree/master/demos/zoa.py)   |