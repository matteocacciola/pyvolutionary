from abc import ABC, abstractmethod
from dataclasses import field
from typing import TypeVar, Any
import numpy as np
from pydantic import BaseModel, model_validator, ConfigDict
from functools import wraps

from .enums import TaskType


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    Especially, it can encode a list of mixed types include integer, float, and string. Better than scikit-learn module.
    """
    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    @staticmethod
    def set_y(y):
        if type(y) not in (list, tuple, np.ndarray):
            y = (y,)
        return y

    def fit(self, y: list | tuple) -> "LabelEncoder":
        """
        Fit label encoder to a given set of labels.
        :param y: labels to encode
        """
        self.unique_labels = sorted(set(y), key=lambda x: (isinstance(x, (int, float)), x))
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        return self

    def transform(self, y: list | tuple) -> list:
        """
        Transform labels to encoded integer labels.
        :param y: labels to encode
        :return: encoded integer labels
        :rtype: list
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.set_y(y)
        return [self.label_to_index[label] for label in y]

    def fit_transform(self, y: list | tuple) -> list:
        """
        Fit label encoder and return encoded labels.
        :param y: target values
        :return: encoded labels
        :rtype: list
        """
        y = self.set_y(y)
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: list | tuple) -> list:
        """
        Transform integer labels to original labels.
        :param y: encoded integer labels
        :return: original labels
        :rtype: list
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.set_y(y)
        return [self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y]


class BaseOptimizationConfig(BaseModel):
    population_size: int
    fitness_error: float | None = None
    max_cycles: int


class Agent(BaseModel):
    position: list[Any]
    cost: float
    fitness: float


class Population(BaseModel):
    agents: list[Agent] = field(default_factory=list)

    # initialize the agents by considering the task type: if it is a minimization task, each agent is accepted as is;
    # otherwise, the position of each agent is multiplied by -1
    def __init__(self, **kwargs: Any):
        def refine_agent(a: Agent, tt: TaskType) -> Agent:
            if tt == TaskType.MIN:
                return a
            # return the agent with the position multiplied by -1
            return a.model_copy(update={"cost": -a.cost})

        task_type = kwargs.get("task_type", TaskType.MIN)
        agents = [refine_agent(a, task_type) for a in kwargs.get("agents", [])]
        kwargs["agents"] = agents
        super().__init__(**kwargs)

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
    def __init__(self, **kwargs: Any):
        def refine_best_solution(a: Agent, tt: TaskType) -> Agent:
            if tt == TaskType.MIN:
                return a
            # return the agent with the position multiplied by -1
            return a.model_copy(update={"cost": -a.cost})

        task_type = kwargs.get("task_type", TaskType.MIN)
        best_solution = kwargs.get("best_solution")
        if best_solution is not None:
            kwargs["best_solution"] = refine_best_solution(best_solution, task_type)
        super().__init__(**kwargs)


class Variable(BaseModel, ABC):
    name: str | None = "var"
    value: Any | None = None

    @abstractmethod
    def randomize(self) -> Any:
        pass

    @abstractmethod
    def get_bounds(self) -> tuple:
        pass

    @abstractmethod
    def correct(self, value) -> Any:
        pass

    @abstractmethod
    def decode(self, value) -> Any:
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

    def get_bounds(self) -> tuple[float, float]:
        return self.lower_bound, self.upper_bound

    def correct(self, value: float | int) -> float:
        self.value = float(np.clip(value, self.lower_bound, self.upper_bound))
        return self.value

    def decode(self, value: float) -> float:
        return value


class DiscreteVariable(Variable):
    choices: list[Any]

    def randomize(self) -> int:
        self.value = np.random.choice(range(0, len(self.choices)))
        return self.value

    def get_bounds(self) -> tuple[int, int]:
        return 0, len(self.choices) - 1

    def correct(self, value: float | int) -> int:
        lb, ub = self.get_bounds()
        self.value = int(np.clip(value, lb, ub))
        return self.value

    def decode(self, value: float | int) -> Any:
        return self.choices[int(value)]


class PermutationVariable(Variable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[Any]
    label_encoder: LabelEncoder = LabelEncoder()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.label_encoder.fit(self.items)

    def randomize(self) -> list[int]:
        self.value = np.random.permutation(range(0, len(self.items))).tolist()
        return self.value

    def get_bounds(self) -> tuple[list, list]:
        n_items = len(self.items)
        lb = np.zeros(n_items)
        ub = (n_items - 1e-4) * np.ones(n_items)
        return lb.tolist(), ub.tolist()

    def correct(self, value: tuple | list | np.ndarray) -> list[int]:
        self.value = np.argsort(value).tolist()
        return self.value

    def decode(self, value: tuple | list | np.ndarray) -> Any:
        value = self.correct(value)
        return self.label_encoder.inverse_transform(value)


class MultiObjectiveVariable(Variable):
    lower_bounds: tuple[float] | list[float]
    upper_bounds: tuple[float] | list[float]

    @model_validator(mode="after")
    def validate_bounds(self) -> "MultiObjectiveVariable":
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Lower and upper bounds must have the same length")
        if np.any(np.array([ub <= lb for lb, ub in zip(self.lower_bounds, self.upper_bounds)])):
            raise ValueError("Upper bound must be greater than lower bound")
        return self

    def randomize(self):
        raise NotImplementedError

    def get_bounds(self) -> tuple[tuple[float] | list[float], tuple[float] | list[float]]:
        return self.lower_bounds, self.upper_bounds

    def correct(self, value: list):
        raise NotImplementedError

    def decode(self, value: list) -> list:
        raise NotImplementedError


def task_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        variables = kwargs.get("variables")
        if not isinstance(variables, list) and not isinstance(variables, MultiObjectiveVariable):
            raise ValueError("Variables must be a list or a multi-objective variable")

        kwargs["is_multi_objective"] = isinstance(variables, MultiObjectiveVariable)
        if kwargs["is_multi_objective"]:
            lower_bounds, upper_bounds = variables.get_bounds()

            variables = [ContinuousVariable(
                name=f"{variables.name}{i}", lower_bound=lb, upper_bound=ub
            ) for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds))]
        kwargs["variables"] = variables
        kwargs["space_dimension"] = len(variables)
        return f(*args, **kwargs)

    return wrapper


class Task(BaseModel, ABC):
    seed: float | None = None
    variables: list[Variable] = field(default_factory=list)
    space_dimension: int
    minmax: TaskType = TaskType.MIN
    data: dict | None = None
    objective_weights: list[float] | None = None
    is_multi_objective: bool = False

    @task_decorator
    def __init__(self, **kwargs: Any):
        tt = kwargs.get("minmax", TaskType.MIN)
        if tt not in TaskType:
            raise ValueError(f"Invalid task type: {tt}")
        kwargs["minmax"] = TaskType(tt)

        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_objective_weights(self) -> "Task":
        if self.objective_weights is None:
            return self
        if not np.all(np.array(self.objective_weights) >= 0):
            raise ValueError("Objective weights must be greater than or equal to zero")
        return self

    @abstractmethod
    def objective_function(self, x: list[float | int]) -> float | list[float]:
        pass

    def transform_position(self, x: list[float | int]) -> dict[str, Any]:
        return {v.name: v.decode(x[i]) for i, v in enumerate(self.variables)}


T = TypeVar("T", Agent, Agent)
