from abc import ABC, abstractmethod
from typing import Generic, Any
import numpy as np
import concurrent.futures as parallel

from .enums import ModeSolver
from .helpers import (
    average_fitness,
    calculate_fitness,
    sort_and_trim,
    sort_by_cost,
    special_agents,
)
from .models import OptimizationResult, Population, T, BaseOptimizationConfig, Task, TaskType, Agent, ContinuousVariable


class OptimizationAbstract(ABC, Generic[T]):
    """
    This class is an abstract for optimization algorithms.
    """

    def __init__(self, config: BaseOptimizationConfig, debug: bool | None = False):
        """
        The constructor of the class.
        :param config: The configuration of the optimization algorithm.
        :param debug: Whether to print debug messages or not.
        """
        self._config = config
        self._debug = debug
        self._mode: ModeSolver = ModeSolver.SERIAL
        self._workers: int = 4
        self._task: Task | None = None
        self._population: list[T] = []
        self._best_agent: T | None = None
        self._worst_agent: T | None = None
        self._cycles = 1

    @abstractmethod
    def optimization_step(self):
        pass

    def before_initialization(self):
        pass

    def after_initialization(self):
        pass

    def _get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the lower and upper bounds of the search space.
        :return: the lower and upper bounds
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        lb, ub = zip(*[
            (v.lower_bound, v.upper_bound) if isinstance(v, ContinuousVariable) else (0, 0)
            for v in self._task.variables
        ])
        return np.array(lb), np.array(ub)

    def _init_position(self, position: list[float] | np.ndarray | None = None) -> list[float]:
        """
        This method initializes the position of the agent of the optimization algorithm. The position is randomly
        generated if it is not provided.
        :param position: the position to initialize
        :return: the initialized position
        :rtype: list[float]
        """
        position = position.tolist() if isinstance(position, np.ndarray) else position
        position = self._correct_position(position if position is not None else self._uniform_position())

        return position

    def _uniform_position(self) -> np.ndarray:
        """
        This method generates a uniform random position in the search space.
        :return: the random position
        :rtype: np.ndarray
        """
        return np.array([v.randomize() for v in self._task.variables])

    def _random_position(self) -> np.ndarray:
        """
        This method generates a random position in the search space.
        :return: the random position
        :rtype: np.ndarray
        """
        lb, _ = self._get_bounds()
        return np.where(
            isinstance(self._task.variables, ContinuousVariable),
            np.random.random() * self._bandwidth() + lb,
            self._uniform_position()
        )

    def _increase_position(self, position: list[float], scale_factor: float | None = None) -> np.ndarray:
        """
        This method increases the position in the search space.
        :param position: the position to increase
        :param scale_factor: the scale factor
        :return: the increased position
        :rtype: np.ndarray
        """
        scale_factor = scale_factor if scale_factor is not None else 1.0
        return np.where(
            isinstance(self._task.variables, ContinuousVariable),
            np.array(position) + self._random_position() / scale_factor,
            self._uniform_position()
        )

    def _uniform_coordinates(self, dimensions: int | list[int]) -> np.ndarray:
        """
        This method generates uniform random coordinates in the search space.
        :param dimensions: the dimensions to generate
        :return: the random coordinates
        :rtype: np.ndarray
        """
        return self._uniform_position()[dimensions]

    def _bandwidth(self) -> np.ndarray:
        """
        This method calculates the bandwidth in the search space for each dimension.
        :return: the bandwidth
        :rtype: np.ndarray
        """
        lb, ub = self._get_bounds()
        return ub - lb

    def _sum_bounds(self) -> np.ndarray:
        """
        This method calculates the sum of the lower and upper bounds in the search space for each dimension.
        :return: the sum of the lower and upper bounds
        :rtype: np.ndarray
        """
        lb, ub = self._get_bounds()
        return lb + ub

    def _fcn(self, x: list[float] | np.ndarray) -> float:
        """
        This method evaluates the objective function.
        :param x: the position to evaluate
        """
        return self._task.objective_function(x) \
            if self._task.minmax == TaskType.MIN else -self._task.objective_function(x)

    def _init_agent(self, position: list[float] | np.ndarray | None = None) -> Agent:
        """
        This method initializes the agent of the optimization algorithm. The position is randomly generated if it is
        not provided. The other properties of the agent.
        """
        position = self._init_position(position)
        cost = self._fcn(position)
        return Agent(position=position, cost=cost, fitness=calculate_fitness(cost, self._task.minmax))

    def _init_population(self):
        """
        This method initializes the population of the optimization algorithm.
        """
        # Serial mode
        if self._mode == ModeSolver.SERIAL:
            self._population = [self._init_agent() for _ in range(0, self._config.population_size)]
            return

        # Parallel mode
        pop = []
        pool = parallel.ThreadPoolExecutor if self._mode == ModeSolver.THREAD else parallel.ProcessPoolExecutor
        with pool(self._workers) as executor:
            executors = [executor.submit(self._init_agent) for _ in range(0, self._config.population_size)]
            for i in parallel.as_completed(executors):
                pop.append(i.result())
        self._population = pop

    def _is_valid_position(self, position: list[float] | np.ndarray) -> bool:
        """
        Check whether the position is valid or not.
        :param position: the position to check
        :return: whether the position is valid or not
        """
        lb, ub = self._get_bounds()
        return np.all(np.less_equal(lb, position)) and np.all(np.less_equal(position, ub))

    def _greedy_select_population(self, new_population: list[T]):
        """
        Perform the greedy selection between the current population and the new one. Both are sorted by cost in
        ascending order. The greedy selection is performed by comparing the costs of each agent in the current
        population with the corresponding agent in the new population. The one with the lowest cost is kept.
        The new population is so created.
        :param new_population: the new population
        """
        sort_by_cost(self._population)
        sort_by_cost(new_population)

        # Serial mode
        if self._mode == ModeSolver.SERIAL:
            self._population = [
                self._greedy_select_agent(agent, new_population[idx]) for idx, agent in enumerate(self._population)
            ]
            return

        # Parallel mode
        pop = []
        pool = parallel.ThreadPoolExecutor if self._mode == ModeSolver.THREAD else parallel.ProcessPoolExecutor
        with pool(self._workers) as executor:
            executors = [executor.submit(
                self._greedy_select_agent, agent, new_population[idx]
            ) for idx, agent in enumerate(self._population)]
            for i in parallel.as_completed(executors):
                pop.append(i.result())
        self._population = pop

    def _greedy_select_agent(self, agent: T, new_agent: T) -> T:
        """
        Perform the greedy selection between the current agent and the new one. The greedy selection is performed by
        comparing the costs of each agent. The one with the lowest cost is kept.
        :param agent: the current agent
        :param new_agent: the new agent
        :return: the best agent
        """
        agent_copy = agent.model_copy()
        return new_agent if new_agent.cost < agent_copy.cost else agent_copy

    def _correct_position(self, position: list[float] | np.ndarray) -> list[Any]:
        """
        Correct the solution if it is outside the bounds by setting the solution to the closest bound. This function is
        used to correct the solution after the position update.
        :param position: the position
        :return: the corrected position
        :rtype: list[Any]
        """
        variables = self._task.variables
        return [
            np.clip(p, variables[idx].lower_bound, variables[idx].upper_bound)
            if isinstance(variables[idx], ContinuousVariable) else p
            for idx, p in enumerate(position)
        ]

    def _extend_and_trim_population(self, new_population: list[T]):
        """
        Extend the population with the new population and trim the population to the population size if the population
        size is exceeded. The population is sorted by cost in ascending order.
        :param new_population: the new population
        """
        self._population.extend(new_population)
        self._population = sort_and_trim(self._population, self._config.population_size)

    def optimize(self, task: Task, mode: str | None = None, workers: int | None = None) -> OptimizationResult:
        """
        This method optimizes the given objective function. At the beginning, a random population is generated and then
        the optimization algorithm is executed. The returned result contains the evolution of the population and the
        best solution found. A list of errors per cycle is also returned in the result itself.
        :param task: the task to optimize
        :param mode, the mode of the solver; possible values are "serial", "thread" and "process"
        :param workers: the number of workers to use, in case of parallel or thread execution
        :return: the result of the optimization
        :rtype: OptimizationResult
        """
        np.random.seed(task.seed)

        evolution: list[Population] = []

        self._task = task
        if workers is not None:
            if workers <= 0:
                raise ValueError("Invalid number of workers. It must be greater than 0")
            self._workers = workers

        if mode is not None:
            try:
                self._mode = ModeSolver(mode)
            except ValueError:
                raise ValueError("Invalid mode. Possible values are 'serial', 'thread' and 'process'")

        self.before_initialization()
        self._init_population()
        self.after_initialization()

        evolution.append(Population(agents=self._population, task_type=task.minmax))
        (self._best_agent, ), (self._worst_agent, ) = special_agents(self._population, n_best=1, n_worst=1)

        errors: list[float] = []
        while True:
            self.optimization_step()
            # append the current population to the evolution, being sure that costs and fitness are updated
            evolution.append(Population(agents=self._population, task_type=task.minmax))

            (self._best_agent, ), (self._worst_agent, ) = special_agents(self._population, n_best=1, n_worst=1)

            # stop when the error is below the error criteria or when the maximum number of cycles is reached
            error, fitness, has_to_stop = self.__should_stop__()
            errors.append(error)
            if self._debug:
                print(f"Cycle {self._cycles} - Best position {self._best_agent.position}, "
                      f"cost {self._best_agent.cost if task.minmax == TaskType.MIN else -self._best_agent.cost} - "
                      f"Average fitness {fitness}, fitness error {error}")
            if has_to_stop:
                break

            self._cycles += 1

        if has_to_stop and self._debug:
            print("Maximum number of cycles reached" if self._cycles >= self._config.max_cycles else
                  f"Error criteria reached - Fitness error: {error}")

        return OptimizationResult(
            evolution=evolution, rates=errors, best_solution=self._best_agent, task_type=task.minmax
        )

    def __should_stop__(self) -> tuple[float, float, bool]:
        """
        Check whether the optimization algorithm has to stop or not based on the error criteria and the current cycle.
        :return: a tuple containing the current error, the average fitness and a boolean indicating whether the
        algorithm has to stop or not
        :rtype: tuple[float, float, bool]
        """
        # Get the optimal population
        fitness_error = self._config.fitness_error
        avg_fit = average_fitness(self._population)
        cycle = self._cycles
        max_cycles = self._config.max_cycles

        current_error = abs(1 - avg_fit)

        # Stop when the error is below the error criteria or when the maximum number of cycles is reached
        if fitness_error is None:
            return current_error, avg_fit, cycle >= max_cycles

        return current_error, avg_fit, current_error <= fitness_error or cycle >= max_cycles
