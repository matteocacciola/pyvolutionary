import math
import random
import numpy as np
from pydantic import BaseModel
import concurrent.futures as parallel

from .enums import ModeSolver
from .models import T, TaskType


def parse_obj_doc(model: BaseModel):
    """
    Parse the docstring of a model and return a string containing the name of each field and its type. The docstring
    must be formatted as follows: "field1 (type1)\nfield2 (type2)\n...". The docstring is used to print the details of
    the model. The details are printed as follows: "field1 (type1)\nfield2 (type2)\n...".
    :param model: the model
    :return: the docstring
    """
    details = model.__annotations__
    doc = [f"{field} ({field_type.__name__})" for field, field_type in details.items()]
    return "\n".join(doc)


def squared_norm(point1: list[float], point2: list[float]):
    """
    Calculate the squared distance between two points in the space. This function is used to avoid the square root
    operation when calculating the distance between two points. The squared distance is calculated as follows:
    squared_distance = sum((point1 - point2)^2)
    :param point1: the first point
    :param point2: the second point
    :return: the squared distance between the two points
    :rtype: float
    """
    return np.sum((np.array(point1) - np.array(point2))**2)


def distance(point1: list[float], point2: list[float]):
    """
    Calculate the distance between two points in the space using the Euclidean distance.
    :param point1: the first point
    :param point2: the second point
    :return: the distance between the two points
    :rtype: float
    """
    return np.sqrt(squared_norm(point1, point2))


def distances(elements: list | np.ndarray) -> np.ndarray:
    """
    Calculate the distance between each element of the provided list
    :param elements: the list of elements
    :return: the distance matrix
    """
    elements = np.array(elements)
    a = elements[:, :-1]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    return np.sqrt(np.einsum('ijk,ijk->ij', b - a, b - a)).squeeze()


def verser(point1: list[float], point2: list[float]) -> np.ndarray:
    """
    Calculate the unit vector between two points in the space. The unit vector is calculated as follows:
    unit_vector = (point1 - point2) / (distance + eps)
    :param point1: the first point
    :param point2: the second point
    :return: the unit vector between the two points
    :rtype: np.ndarray
    """
    dist = distance(point1, point2)
    return (np.array(point1) - np.array(point2)) / (dist + np.finfo(float).eps)


def calculate_fitness(value: float, task_type: TaskType) -> float:
    """
    Calculate the fitness of the agent based on its cost value. The fitness is calculated as follows:
    fitness = 1 / (1 + cost) if cost >= 0 else 1 + abs(cost) (the higher the fitness, the best the agent).
    :param value: the cost value
    :param task_type: the type of task (minimization or maximization)
    :return: the fitness value
    :rtype: float
    """
    value = value if task_type == TaskType.MIN else -value
    return (1 / (value + 1)) if value >= 0 else (1 + abs(value))


def sort_by_cost(population: list[T], task_type: TaskType | None = TaskType.MIN) -> list[T]:
    """
    Sort the population by the cost value of each agent (the lower the cost, the best the agent). The sorting is done
    in-place. If task_type is 'max', the sorting is done in descending order (the higher the cost, the worst the agent).
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the sorted population
    :rtype: list[Agent]
    """
    pop_new = population.copy()
    pop_new.sort(key=lambda agent: agent.cost, reverse=(task_type == TaskType.MAX))
    return pop_new


def sort_by_cost_indexes(population: list[T], task_type: TaskType | None = TaskType.MIN) -> list[int]:
    """
    Sort the population by the cost value of each agent (the lower the cost, the best the agent). The sorting is done
    in-place. If task_type is 'max', the sorting is done in descending order (the higher the cost, the worst the agent).
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the indexes of the sorted population
    :rtype: list[int]
    """
    result = np.argsort([agent.cost for agent in population], axis=0)
    if task_type == TaskType.MAX:
        result = result[::-1]
    return result.tolist()


def sort_and_trim(population: list[T], population_size: int) -> list[T]:
    """
    Sort the population according to the cost in ascending order and trim the population to the population size if
    the population size is exceeded.
    :param population: the current population
    :param population_size: the population size
    :return: the sorted and trimmed population
    :rtype: list[Agent]
    """
    return sort_by_cost(population)[:population_size]


def best_agent(population: list[T], task_type: TaskType | None = TaskType.MIN) -> T:
    """
    Get the best agent of the population based on its cost value (the lower the cost, the best the agent).
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the best agent
    :rtype: Agent
    """
    b_agent, = best_agents(population, 1, task_type)
    return b_agent


def best_agent_formatted(population: list[T], n_params: int, task_type: TaskType | None = TaskType.MIN) -> T:
    """
    Get the best agent of the population based on its cost value (the lower the cost, the best the agent).
    The positions of the best agent will be splitted into n_params parts.
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the best agent
    :rtype: Agent
    """
    b_agent = best_agent(population, task_type)

    best_agent_position = np.array(b_agent.position)  # Ensure it's a NumPy array
    if len(best_agent_position) % n_params != 0:
        raise ValueError("Length of `best_agent.position` must be a multiple of `n_params`.")

    # Reshape the position
    b_agent.position = best_agent_position.reshape(-1, n_params)

    return b_agent


def best_agent_index(population: list[T], task_type: TaskType | None = TaskType.MIN) -> int:
    """
    Get the index of the best agent of the population based on its cost value (the lower the cost, the best the agent).
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the index of the best agent
    :rtype: int
    """
    return best_agents_indexes(population, 1, task_type)[0]


def worst_agent(population: list[T], task_type: TaskType | None = TaskType.MIN) -> T:
    """
    Get the worst agent of the population based on its cost value (the higher the cost, the worst the agent).
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the worst agent
    :rtype: Agent
    """
    w_agent, = worst_agents(population, 1, task_type)
    return w_agent


def worst_agent_formatted(population: list[T], n_params: int, task_type: TaskType | None = TaskType.MIN) -> T:
    """
    Get the worst agent of the population based on its cost value (the lower the cost, the best the agent).
    The positions of the worst agent will be splitted into n_params parts.
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the best agent
    :rtype: Agent
    """
    w_agent = worst_agent(population, task_type)

    worst_agent_position = np.array(w_agent.position)  # Ensure it's a NumPy array
    if len(worst_agent_position) % n_params != 0:
        raise ValueError("Length of `worst_agent.position` must be a multiple of `n_params`.")

    # Reshape the position
    w_agent.position = worst_agent_position.reshape(-1, n_params)

    return w_agent


def worst_agent_index(population: list[T], task_type: TaskType | None = TaskType.MIN) -> int:
    """
    Get the index of the worst agent of the population based on its cost value (the lower the cost, the best the agent).
    :param population: the population
    :param task_type: the type of task (minimization or maximization)
    :return: the index of the worst agent
    :rtype: int
    """
    return worst_agents_indexes(population, 1, task_type)[0]


def best_agents(population: list[T], n_best: int | None = 3, task_type: TaskType | None = TaskType.MIN) -> list[T]:
    """
    Get the best agents of the population based on their cost value (the lower the cost, the best the agent).
    :param population: the population
    :param n_best: the number of best agents to return
    :param task_type: the type of task (minimization or maximization)
    :return: the best agents
    :rtype: list[T]
    """
    return sort_by_cost(population, task_type=task_type)[:n_best]


def best_agents_indexes(
    population: list[T], n_best: int | None = 3, task_type: TaskType | None = TaskType.MIN
) -> list[int]:
    """
    Get the indexes of the best agents of the population based on their cost value (the lower the cost, the best the
    agent).
    :param population: the population
    :param n_best: the number of best agents to return
    :param task_type: the type of task (minimization or maximization)
    :return: the indexes of the best agents
    :rtype: list[int]
    """
    return sort_by_cost_indexes(population, task_type=task_type)[:n_best]


def worst_agents(population: list[T], n_worst: int | None = 3, task_type: TaskType | None = TaskType.MIN) -> list[T]:
    """
    Get the worst agents of the population based on their cost value (the higher the cost, the worst the agent).
    :param population: the population
    :param n_worst: the number of worst agents to return
    :param task_type: the type of task (minimization or maximization)
    :return: the worst agents
    :rtype: list[T]
    """
    return sort_by_cost(population, task_type=task_type)[len(population)-n_worst:]


def worst_agents_indexes(population: list[T], n_worst: int | None = 3, task_type: TaskType | None = TaskType.MIN) -> list[int]:
    """
    Get the indexes of the worst agents of the population based on their cost value (the higher the cost, the worst the
    agent).
    :param population: the population
    :param n_worst: the number of worst agents to return
    :param task_type: the type of task (minimization or maximization)
    :return: the indexes of the worst agents
    :rtype: list[int]
    """
    return sort_by_cost_indexes(population, task_type=task_type)[len(population)-n_worst:]


def special_agents(
    population: list[T],
    n_best: int | None = None,
    n_worst: int | None = None,
    task_type: TaskType | None = TaskType.MIN
) -> tuple[list[T], list[T]]:
    """
    Get the best and worst agents of the population based on their cost value (the lower the cost, the best the agent).
    Either n_best or n_worst must be provided. The best and worst agents are returned as a tuple.
    :param population: the population
    :param n_best: the number of best agents to return
    :param n_worst: the number of worst agents to return
    :param task_type: the type of task (minimization or maximization)
    :return: a tuple containing the best and worst agents
    :rtype: tuple[list[T], list[T]]
    """
    if n_best is None and n_worst is None:
        raise ValueError("Either n_best or n_worst must be provided")

    best = []
    if n_best is not None:
        best = best_agents(population, n_best, task_type)

    worst = []
    if n_worst is not None:
        worst = worst_agents(population, n_worst, task_type)

    return best, worst


def average_fitness(population: list[T]) -> float:
    """
    Calculate the average fitness of the population based on the fitness of each agent of the population itself.
    :param population:
    :return: the average fitness
    :rtype: float
    """
    return np.average([agent.fitness for agent in population])


def get_partner_index(index: int, num_elements: int) -> int:
    while True:
        partner_index = random.randint(0, num_elements - 1)
        if partner_index != index:
            break
    return partner_index


def roulette_wheel_indexes(probabilities: np.ndarray, num: int | None = 1) -> list[int]:
    """
    Select an index using roulette wheel selection with probabilities as weights of the selection process and return the
    selected index of the population array of the algorithm instance.
    :param probabilities: the probabilities of each element of the population
    :param num: the number of indexes to return
    :return: the indexes of the selected elements
    :rtype: int
    """
    final_probabilities = np.max(probabilities) - probabilities
    k = list(set(range(0, len(probabilities))))
    if all(final_probabilities == 0):
        return np.random.choice(k, size=num, replace=False)

    p = final_probabilities / np.sum(final_probabilities)
    return np.random.choice(k, size=num, replace=len([i for i in p if i != 0]) < num, p=p)


def random_selection(p: list | np.ndarray) -> int:
    """
    Randomly select an index from the provided probabilities array.
    :param p: the probabilities array
    :return: the selected index
    :rtype: int
    """
    r = np.random.random()
    c = np.cumsum(p)
    index = [i for i, x in enumerate(c) if r <= x]
    return index[0]


def get_levy_flight_step(
    beta: float = 1.0,
    multiplier: float = 0.001,
    size: int | list | tuple | np.ndarray = None,
    case: int = 0
) -> float | list | np.ndarray:
    """
    Get the Levy-flight step size
    :param beta: Should be in range [0, 2]. 0-1: small range --> exploit. 1-2: large range --> explore
    :param multiplier: default = 0.001
    :param size: size of levy-flight steps, for example: (3, 2), 5, (4, )
    :param case: Should be one of these value [0, 1, -1]. 0: return multiplier * s * self.generator.uniform().
    1: return multiplier * s * self.generator.normal(0, 1). -1: return multiplier * s
    :return: the step size of Levy-flight trajectory
    :rtype: float, list, np.ndarray
    """
    # u and v are two random variables which follow self.generator.normal distribution
    # sigma_u : standard deviation of u
    sigma_u = np.power(math.gamma(1. + beta) * np.sin(np.pi * beta / 2) / (
        math.gamma((1 + beta) / 2.) * beta * np.power(2., (beta - 1) / 2)
    ), 1. / beta)
    size = 1 if size is None else size
    u = np.random.normal(0, sigma_u ** 2, size)
    v = np.random.normal(0, 1, size)
    s = u / np.power(np.abs(v), 1 / beta)

    step = multiplier * s
    if case == 0:
        step = multiplier * s * np.random.uniform()
    elif case == 1:
        step = multiplier * s * np.random.normal(0, 1)
    return step[0] if size == 1 else step


def get_pool_executor(mode: ModeSolver, n_workers: int = None) -> parallel.Executor:
    """
    Get the executor of the provided mode.
    :param mode: the mode
    :param n_workers: the number of workers
    :return: the executor
    :rtype: parallel.Executor
    """
    return (
        parallel.ThreadPoolExecutor(n_workers) if mode == ModeSolver.THREAD else parallel.ProcessPoolExecutor(n_workers)
    )


def get_pool_results(executors: list[parallel.Future]) -> list:
    """
    Get the results of the provided executors.
    :param executors: the executors
    :return: the results
    :rtype: list
    """
    res = []
    for i in parallel.as_completed(executors):
        res.append(i.result())
    return res


def find_centers(pop_groups: list[list[T]]) -> list[T]:
    return [best_agent(pop_group).model_copy() for pop_group in pop_groups]


def runge_kutta(xb: np.ndarray, xw: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
    dim = len(xb)
    C = np.random.randint(1, 3) * (1 - np.random.random())
    r1 = np.random.random(dim)
    r2 = np.random.random(dim)
    K1 = 0.5 * (np.random.random() * xw - C * xb)
    K2 = 0.5 * (np.random.random() * (xw + r2 * K1 * delta_x / 2) - (C * xb + r1 * K1 * delta_x / 2))
    K3 = 0.5 * (np.random.random() * (xw + r2 * K2 * delta_x / 2) - (C * xb + r1 * K2 * delta_x / 2))
    K4 = 0.5 * (np.random.random() * (xw + r2 * K3 * delta_x) - (C * xb + r1 * K3 * delta_x))
    return (K1 + 2 * K2 + 2 * K3 + K4) / 6
