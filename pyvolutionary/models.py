from abc import ABC, abstractmethod
from dataclasses import field
from typing import TypeVar
from typing import Any
import numpy as np
from pydantic import BaseModel, field_validator, model_validator, ConfigDict, PrivateAttr

from .enums import TaskType


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    Especially, it can encode a list of mixed types include integer, float, and string. Better than scikit-learn module.
    """
    def __init__(self):
        self.__unique_labels__ = None
        self.__label_to_index__ = {}

    @staticmethod
    def __set_y__(y):
        if type(y) not in (list, tuple, np.ndarray):
            y = (y,)
        return y

    def fit(self, y: list | tuple) -> "LabelEncoder":
        """
        Fit label encoder to a given set of labels.
        :param y: labels to encode
        """
        self.__unique_labels__ = sorted(set(y), key=lambda x: (isinstance(x, (int, float)), x))
        self.__label_to_index__ = {label: i for i, label in enumerate(self.__unique_labels__)}
        return self

    def transform(self, y: list | tuple) -> list:
        """
        Transform labels to encoded integer labels.
        :param y: labels to encode
        :return: encoded integer labels
        :rtype: list
        """
        if self.__unique_labels__ is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.__set_y__(y)
        return [self.__label_to_index__[label] for label in y]

    def fit_transform(self, y: list | tuple) -> list:
        """
        Fit label encoder and return encoded labels.
        :param y: target values
        :return: encoded labels
        :rtype: list
        """
        y = self.__set_y__(y)
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: list | tuple) -> list:
        """
        Transform integer labels to original labels.
        :param y: encoded integer labels
        :return: original labels
        :rtype: list
        """
        if self.__unique_labels__ is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.__set_y__(y)
        return [self.__unique_labels__[i] if i in self.__label_to_index__.values() else "unknown" for i in y]


class EarlyStopping(BaseModel):
    patience: int | None = 1
    min_delta: float | None = 1e-4

    @field_validator("patience")
    def validate_patience(cls, v):
        if v is not None and v < 1:
            raise ValueError(f"\"patience\" must be greater than or equal to one. Got {v}")
        return v


class BaseOptimizationConfig(BaseModel):
    population_size: int
    fitness_error: float | None = 0.1
    max_cycles: int
    early_stopping: EarlyStopping | None = None


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

    @abstractmethod
    def get(self) -> Any:
        pass

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

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def has_children(self) -> bool:
        pass


class ContinuousVariable(Variable):
    lower_bound: float
    upper_bound: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContinuousVariable":
        if self.upper_bound <= self.lower_bound:
            raise ValueError("Upper bound must be greater than lower bound")
        return self

    def get(self) -> "ContinuousVariable":
        return self

    def randomize(self) -> float:
        return np.random.uniform(self.lower_bound, self.upper_bound)

    def get_bounds(self) -> tuple[float, float]:
        return self.lower_bound, self.upper_bound

    def correct(self, value: float | int) -> float:
        return float(np.clip(value, self.lower_bound, self.upper_bound))

    def decode(self, value: float) -> float:
        return value

    def size(self) -> int:
        return 1

    def has_children(self) -> bool:
        return False


class ContinuousMultiVariable(Variable):
    lower_bounds: list[float]
    upper_bounds: list[float]

    _children: list[ContinuousVariable] = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        lower_bounds, upper_bounds = self.get_bounds()
        self._children = [ContinuousVariable(
            name=f"{self.name}{i}", lower_bound=lb, upper_bound=ub
        ) for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds))]

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContinuousMultiVariable":
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Lower and upper bounds must have the same length")
        if np.any(np.array([ub <= lb for lb, ub in zip(self.lower_bounds, self.upper_bounds)])):
            raise ValueError("Upper bound must be greater than lower bound")
        return self

    def get(self) -> list["ContinuousVariable"]:
        return self._children

    def randomize(self):
        return [v.randomize() for v in self._children]

    def get_bounds(self) -> tuple[tuple[float] | list[float], tuple[float] | list[float]]:
        return self.lower_bounds, self.upper_bounds

    def correct(self, value: list):
        return [v.correct(value[idx]) for idx, v in enumerate(self._children)]

    def decode(self, value: list) -> list:
        return [v.decode(value[idx]) for idx, v in enumerate(self._children)]

    def size(self) -> int:
        return len(self.lower_bounds)

    def has_children(self) -> bool:
        return True


class DiscreteVariable(Variable):
    choices: list[Any]

    def get(self) -> "DiscreteVariable":
        return self

    def randomize(self) -> int:
        return np.random.choice(range(0, len(self.choices)))

    def get_bounds(self) -> tuple[int, int]:
        return 0, len(self.choices) - 1

    def correct(self, value: float | int) -> int:
        lb, ub = self.get_bounds()
        return int(np.clip(value, lb, ub))

    def decode(self, value: float | int) -> Any:
        return self.choices[int(value)]

    def size(self) -> int:
        return 1

    def has_children(self) -> bool:
        return False


class DiscreteMultiVariable(Variable):
    choices: list[list[Any]]

    _children: list[DiscreteVariable] = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._children = [
            DiscreteVariable(name=f"{self.name}{i}", choices=self.choices[i]) for i in range(len(self.choices))
        ]

    def get(self) -> list["DiscreteVariable"]:
        return self._children

    def randomize(self):
        return [v.randomize() for v in self._children]

    def get_bounds(self) -> list[tuple[int, int]]:
        return [v.get_bounds() for v in self._children]

    def correct(self, value: list):
        return [v.correct(value[idx]) for idx, v in enumerate(self._children)]

    def decode(self, value: list) -> list:
        return [v.decode(value[idx]) for idx, v in enumerate(self._children)]

    def size(self) -> int:
        return len(self.choices)

    def has_children(self) -> bool:
        return True


class PermutationVariable(Variable):
    items: list[Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _label_encoder: LabelEncoder = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self.items)

    def get(self) -> "PermutationVariable":
        return self

    def randomize(self) -> list[int]:
        return np.random.permutation(range(0, len(self.items))).tolist()

    def get_bounds(self) -> tuple[list, list]:
        n_items = len(self.items)
        lb = np.zeros(n_items)
        ub = (n_items - 1e-4) * np.ones(n_items)
        return lb.tolist(), ub.tolist()

    def correct(self, value: tuple | list | np.ndarray) -> list[int]:
        return np.argsort(value).tolist()

    def decode(self, value: tuple | list | np.ndarray) -> Any:
        value = self.correct(value)
        return self._label_encoder.inverse_transform(value)

    def size(self) -> int:
        return 1

    def has_children(self) -> bool:
        return False


class MultiObjectiveVariable(Variable):
    lower_bounds: tuple[float] | list[float]
    upper_bounds: tuple[float] | list[float]

    _children: list[ContinuousVariable] = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        lower_bounds, upper_bounds = self.get_bounds()
        self._children = [ContinuousVariable(
            name=f"{self.name}{i}", lower_bound=lb, upper_bound=ub
        ) for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds))]

    @model_validator(mode="after")
    def validate_bounds(self) -> "MultiObjectiveVariable":
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Lower and upper bounds must have the same length")
        if np.any(np.array([ub <= lb for lb, ub in zip(self.lower_bounds, self.upper_bounds)])):
            raise ValueError("Upper bound must be greater than lower bound")
        return self

    def get(self) -> list["ContinuousVariable"]:
        return self._children

    def randomize(self):
        return [v.randomize() for v in self._children]

    def get_bounds(self) -> tuple[tuple[float] | list[float], tuple[float] | list[float]]:
        return self.lower_bounds, self.upper_bounds

    def correct(self, value: list):
        return [v.correct(value[idx]) for idx, v in enumerate(self._children)]

    def decode(self, value: list) -> list:
        return [v.decode(value[idx]) for idx, v in enumerate(self._children)]

    def size(self) -> int:
        return len(self.lower_bounds)

    def has_children(self) -> bool:
        return True


class BinaryVariable(Variable):
    n_vars: int

    _children: list[DiscreteVariable] = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._children = [DiscreteVariable(name=f"{self.name}{i}", choices=[0, 1]) for i in range(self.n_vars)]

    @field_validator("n_vars")
    def validate_n_vars(cls, v):
        if v <= 0:
            raise ValueError(f"\"n_vars\" must be greater than zero. Got {v}")
        return v

    def get(self) -> list["DiscreteVariable"]:
        return self._children

    def randomize(self):
        return [v.randomize() for v in self._children]

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.n_vars)
        ub = (2 - np.finfo(float).eps) * np.ones(self.n_vars)
        return lb, ub

    def correct(self, value: list):
        return [v.correct(value[idx]) for idx, v in enumerate(self._children)]

    def decode(self, value: list) -> list:
        return [v.decode(value[idx]) for idx, v in enumerate(self._children)]

    def size(self) -> int:
        return self.n_vars

    def has_children(self) -> bool:
        return True


class Task(BaseModel, ABC):
    seed: float | None = None
    variables: list[Variable] = field(default_factory=list)
    space_dimension: int
    minmax: TaskType = TaskType.MIN
    data: dict | None = None
    objective_weights: list[float] | None = None

    _EPS = PrivateAttr()

    def __init__(self, **kwargs: Any):
        variables = kwargs.get("variables")
        kwargs["space_dimension"] = sum([v.size() for v in variables])
        super().__init__(**kwargs)

        self._EPS = np.finfo(float).eps

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

    def get_variables(self) -> list[Variable]:
        return [item for v in self.variables for item in (v.get() if v.has_children() else [v.get()])]

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the lower and upper bounds of the search space.
        :return: the lower and upper bounds
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        lb = []
        ub = []
        for v in self.variables:
            lb_, ub_ = v.get_bounds()
            lb.extend(lb_ if v.has_children() else [lb_])
            ub.extend(ub_ if v.has_children() else [ub_])

        return np.array(lb), np.array(ub)

    def correct_solution(self, solution: list[float | int] | np.ndarray) -> list[float | int]:
        """
        Correct the solution if it is outside the bounds by setting the solution to the closest bound. This function is
        used to correct the solution after the solution update.
        :param solution: the solution
        :return: the corrected solution
        :rtype: list[Any]
        """
        variables = self.get_variables()
        return [v.correct(c) for c, v in zip(solution, variables)]

    def empty_solution(self) -> list[float]:
        """
        This method generates a uniform random solution in the search space.
        :return: the random solution
        :rtype: list[float]
        """
        solution = [item for v in self.variables for item in (v.randomize() if v.has_children() else [v.randomize()])]
        return solution

    def initial_solution(self, solution: list[float] | np.ndarray | None = None) -> list[float]:
        """
        This method initializes the solution of the agent of the optimization algorithm. The solution is randomly
        generated if it is not provided.
        :param solution: the solution to initialize
        :return: the initialized solution
        :rtype: list[float]
        """
        solution = solution.tolist() if isinstance(solution, np.ndarray) else solution
        return self.correct_solution(solution if solution is not None else self.empty_solution())

    def amend_solution(self, solution: list[float | int] | np.ndarray) -> np.ndarray:
        solution = solution if isinstance(solution, np.ndarray) else np.array(solution)
        lb, ub = self.get_bounds()
        return np.where(np.logical_and(lb <= solution <= ub), solution, np.array(self.initial_solution()))

    def random_solution(self) -> list[float]:
        """
        This method generates a random solution in the search space.
        :return: the random solution
        :rtype: list[float]
        """
        lb, _ = self.get_bounds()
        variables = self.get_variables()
        return np.where(
            isinstance(variables, ContinuousVariable), np.random.random() * self.bandwidth() + lb, self.empty_solution()
        ).tolist()

    def increase_solution(self, solution: list[float], scale_factor: float | None = None) -> np.ndarray:
        """
        This method increases the solution in the search space.
        :param solution: the solution to increase
        :param scale_factor: the scale factor
        :return: the increased solution
        :rtype: np.ndarray
        """
        scale_factor = scale_factor if scale_factor is not None else 1.0
        variables = self.get_variables()
        return np.where(
            isinstance(variables, ContinuousVariable),
            np.array(solution) + np.array(self.random_solution()) / scale_factor,
            self.empty_solution()
        )

    def uniform_coordinates(self, dimensions: int | list[int]) -> list[float]:
        """
        This method generates uniform random coordinates in the search space.
        :param dimensions: the dimensions to generate
        :return: the random coordinates
        :rtype: list[float]
        """
        return np.array(self.empty_solution())[dimensions].tolist()

    def bandwidth(self) -> np.ndarray:
        """
        This method calculates the bandwidth in the search space for each dimension.
        :return: the bandwidth
        :rtype: np.ndarray
        """
        lb, ub = self.get_bounds()
        return ub - lb

    def sum_bounds(self) -> np.ndarray:
        """
        This method calculates the sum of the lower and upper bounds in the search space for each dimension.
        :return: the sum of the lower and upper bounds
        :rtype: np.ndarray
        """
        lb, ub = self.get_bounds()
        return lb + ub

    def is_valid_solution(self, solution: list[float] | np.ndarray) -> bool:
        """
        Check whether the solution is valid or not.
        :param solution: the solution to check
        :return: whether the solution is valid or not
        """
        lb, ub = self.get_bounds()
        return np.all(np.less_equal(lb, solution)) and np.all(np.less_equal(solution, ub))

    def solve(self, x: list[float | int]) -> float | list[float]:
        solution = self.correct_solution(x)
        return self.objective_function(solution)

    def transform_solution(self, x: list[float | int]) -> dict[str, Any]:
        if len(x) == 1:
            return {self.variables[0].name: self.variables[0].decode(x[0])}
        counter = 0
        solution = {}
        for v in self.variables:
            temp = x[counter:(counter + v.size())]
            solution[v.name] = v.decode(temp if len(temp) > 1 else temp[0])
            counter += v.size()
        return solution

    @property
    def name(self):
        return self.__class__.__name__


T = TypeVar("T", Agent, Agent)
