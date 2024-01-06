from abc import ABC, abstractmethod
from dataclasses import field
from typing import TypeVar, Any
import numpy as np
from pydantic import BaseModel, model_validator

from .enums import TaskType


class BaseOptimizationConfig(BaseModel):
    population_size: int
    fitness_error: float | None = None
    max_cycles: int


class Agent(BaseModel):
    position: list[float]
    cost: float
    fitness: float


class Population(BaseModel):
    agents: list[Agent] = field(default_factory=list)

    # initialize the agents by considering the task type: if it is a minimization task, each agent is accepted as is;
    # otherwise, the position of each agent is multiplied by -1
    def __init__(self, **data: Any):
        def refine_agent(a: Agent, tt: TaskType) -> Agent:
            if tt == TaskType.MIN:
                return a
            # return the agent with the position multiplied by -1
            return a.model_copy(update={"cost": -a.cost})

        task_type = data.get("task_type", TaskType.MIN)
        agents = [refine_agent(a, task_type) for a in data.get("agents", [])]
        data["agents"] = agents
        super().__init__(**data)

    def __iter__(self):
        return iter(self.agents)

    def __getitem__(self, item):
        return self.agents[item]

    def __setitem__(self, key, value):
        self.agents[key] = value

    def __len__(self):
        return len(self.agents)

    def __str__(self):
        return str(self.agents)


class OptimizationResult(BaseModel):
    evolution: list[Population] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    best_solution: Agent | None = None

    # initialize the best_solution by considering the task type: if it is a minimization task, it is accepted as is;
    # otherwise, the position is multiplied by -1
    def __init__(self, **data: Any):
        def refine_best_solution(a: Agent, tt: TaskType) -> Agent:
            if tt == TaskType.MIN:
                return a
            # return the agent with the position multiplied by -1
            return a.model_copy(update={"cost": -a.cost})

        task_type = data.get("task_type", TaskType.MIN)
        best_solution = data.get("best_solution")
        if best_solution is not None:
            data["best_solution"] = refine_best_solution(best_solution, task_type)
        super().__init__(**data)


class Variable(BaseModel, ABC):
    name: str
    value: Any | None = None

    @abstractmethod
    def randomize(self) -> None:
        pass


class ContinuousVariable(Variable):
    lower_bound: float
    upper_bound: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContinuousVariable":
        if self.upper_bound <= self.lower_bound:
            raise ValueError("Upper bound must be greater than lower bound")
        return self

    def randomize(self) -> float:
        self.value = np.random.uniform(self.lower_bound, self.upper_bound)
        return self.value


class DiscreteVariable(Variable):
    choices: list[Any]

    def randomize(self) -> int:
        self.value = np.random.choice(range(0, len(self.choices)))
        return self.value


class Task(BaseModel, ABC):
    seed: float | None = None
    variables: list[Variable] = field(default_factory=list)
    space_dimension: int
    minmax: TaskType = TaskType.MIN

    def __init__(self, **data: Any):
        variables = data.get("variables", [])
        data["space_dimension"] = len(variables)
        super().__init__(**data)

    @abstractmethod
    def objective_function(self, x: list[float | int]) -> float:
        pass

    def transform_position(self, x: list[float | int]) -> dict[str, Any]:
        return {
            v.name: x[i] if isinstance(v, ContinuousVariable) else v.choices[int(x[i])]
            for i, v in enumerate(self.variables)
        }


T = TypeVar("T", Agent, Agent)
