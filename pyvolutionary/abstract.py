from abc import ABC, abstractmethod
from typing import Generic, Final, Any
import numpy as np

from .enums import ModeSolver
from .helpers import (
    average_fitness,
    calculate_fitness,
    sort_and_trim,
    sort_by_cost,
    special_agents,
    get_pool_executor,
    get_pool_results,
)
from .models import OptimizationResult, Population, T, BaseOptimizationConfig, Task, TaskType, Agent


class OptimizationAbstract(ABC, Generic[T]):
    """
    This class is an abstract for optimization algorithms.
    """
    EPS: Final[float] = np.finfo(float).eps

    def __init__(self, config: BaseOptimizationConfig | None = None, debug: bool | None = False):
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
        self._current_cycle = 1
        self._errors = []
        self._error_diffs = []

    @abstractmethod
    def optimization_step(self):
        pass

    @abstractmethod
    def set_config_parameters(self, parameters: dict[str, Any]):
        pass

    def before_initialization(self):
        pass

    def after_initialization(self):
        pass

    @property
    def configuration(self) -> BaseOptimizationConfig | None:
        """
        This property returns the configuration of the optimization algorithm.
        :return: the configuration
        :rtype: BaseOptimizationConfig
        """
        return self._config

    @property
    def name(self):
        return self.__class__.__name__

    def _fcn(self, x: list[float] | np.ndarray) -> float | list[float]:
        """
        This method evaluates the objective function.
        :param x: the position to evaluate
        :return the cost of the position, or the list of costs if the objective function is multi-objective
        :rtype: float | list[float]
        """
        return self._task.solve(x) if self._task.minmax == TaskType.MIN else -1 * self._task.solve(x)

    def _init_agent(self, position: list[Any] | np.ndarray | None = None) -> Agent:
        """
        This method initializes the agent of the optimization algorithm. The position is randomly generated if it is
        not provided. The other properties of the agent.
        """
        position = self._task.initial_solution(position)
        cost = self._fcn(position)
        n_weights = len(self._task.objective_weights) if self._task.objective_weights is not None else 1
        n_objectives = len(cost) if isinstance(cost, list) else 1
        if n_weights != n_objectives:
            raise ValueError(f"Invalid number of weights. Expected {n_weights}, found {n_objectives}")

        cost = np.dot(cost, self._task.objective_weights) if self._task.objective_weights is not None else cost
        return Agent(position=position, cost=cost, fitness=calculate_fitness(cost, self._task.minmax))
    
    def _generate_agents(self, n_agents: int) -> list[Agent]:
        """
        This method initializes a number of agents of the optimization algorithm.
        """
        # Serial mode
        if self._mode == ModeSolver.SERIAL:
            return [self._init_agent() for _ in range(0, n_agents)]

        # Parallel mode
        with get_pool_executor(self._mode, self._workers) as executor:
            executors = [executor.submit(self._init_agent) for _ in range(0, n_agents)]
            pop = get_pool_results(executors)
        return pop

    def _init_population(self):
        """
        This method initializes the population of the optimization algorithm.
        """
        self._population = self._generate_agents(self._config.population_size)

    def _greedy_select_population(self, new_population: list[T]):
        """
        Perform the greedy selection between the current population and the new one. Both are sorted by cost in
        ascending order. The greedy selection is performed by comparing the costs of each agent in the current
        population with the corresponding agent in the new population. The one with the lowest cost is kept.
        The new population is so created.
        :param new_population: the new population
        """
        self._population = sort_by_cost(self._population)
        new_population = sort_by_cost(new_population)

        # Serial mode
        if self._mode == ModeSolver.SERIAL:
            self._population = [
                self._greedy_select_agent(agent, new_population[idx]) for idx, agent in enumerate(self._population)
            ]
            return

        # Parallel mode
        with get_pool_executor(self._mode, self._workers) as executor:
            executors = [executor.submit(
                self._greedy_select_agent, agent, new_population[idx]
            ) for idx, agent in enumerate(self._population)]
            self._population = get_pool_results(executors)

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

    def _extend_and_trim_population(self, new_population: list[T]):
        """
        Extend the population with the new population and trim the population to the population size if the population
        size is exceeded. The population is sorted by cost in ascending order.
        :param new_population: the new population
        """
        if len(new_population) == 0:
            return
        self._population.extend(new_population)
        self._population = sort_and_trim(self._population, self._config.population_size)

    def _replace_and_trim_population(self, new_population: list[T]):
        """
        Extend the population with the new population and trim the population to the population size if the population
        size is exceeded. The population is sorted by cost in ascending order.
        :param new_population: the new population
        """
        self._population = sort_and_trim(new_population, self._config.population_size)

    def _generate_group_population(
        self, n_groups: int, n_agents: int, with_residual: bool | None = True
    ) -> list[list[T]]:
        """
        Generate a list of group population from the current population. The population is divided into n_groups
        groups, each composed by n_agents agents. The residual agents are added to the last group.
        :param n_groups: the number of groups
        :param n_agents: the number of agents in each group
        :param with_residual: whether to add the residual agents to the last group or not (default: True)
        :return: a list of group population
        :rtype: list[list[Agent]]
        """
        # calculate the groups composed by n_agents
        groups = []
        for idx in range(0, n_groups):
            group = self._population[idx * n_agents:(idx + 1) * n_agents]
            groups.append([agent.model_copy() for agent in group])

        if not with_residual:
            return groups

        # calculate the group composed by the residual agents
        residual = self._config.population_size % n_groups
        if residual != 0:
            groups.append([agent.model_copy() for agent in self._population[-residual:]])
        return groups

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
        if not self._config:
            raise ValueError("Invalid configuration")

        np.random.seed(task.seed)
        evolution: list[Population] = []

        if workers is not None:
            if workers <= 0:
                raise ValueError("Invalid number of workers. It must be greater than 0")
            self._workers = workers

        if mode is not None:
            try:
                self._mode = ModeSolver(mode)
            except ValueError:
                raise ValueError("Invalid mode. Possible values are \"serial\", \"thread\" and \"process\"")

        self._task = task

        self.before_initialization()

        if self._debug:
            print(f"Starting optimization with {self._config.population_size} agents, "
                  f"{self._config.max_cycles} cycles and {self._workers} workers")
            print(f"Task: {self._task.name} - Minmax: {self._task.minmax} - ")
            print("Initializing population...")

        self._init_population()
        evolution.append(Population(agents=self._population, task_type=task.minmax))
        (self._best_agent, ), (self._worst_agent, ) = special_agents(self._population, n_best=1, n_worst=1)

        self.after_initialization()

        if self._debug:
            print("Population initialized")

        while True:
            if self._debug:
                print(f"Starting cycle {self._current_cycle}...")

            self.optimization_step()
            # append the current population to the evolution, being sure that costs and fitness are updated
            evolution.append(Population(agents=self._population, task_type=task.minmax))

            (self._best_agent, ), (self._worst_agent, ) = special_agents(self._population, n_best=1, n_worst=1)

            # stop when the error is below the error criteria or when the maximum number of cycles is reached
            error, fitness, has_to_stop = self.__error_check__()
            if self._debug:
                print(f"Cycle {self._current_cycle} completed - Best position {self._best_agent.position}, "
                      f"cost {self._best_agent.cost if task.minmax == TaskType.MIN else -self._best_agent.cost} - "
                      f"Average fitness {fitness}, fitness error {error}")
            if has_to_stop:
                break

            self._current_cycle += 1

        if has_to_stop and self._debug:
            print("Maximum number of cycles reached" if self._current_cycle >= self._config.max_cycles else
                  f"Error criteria reached - Fitness error: {error}")

        return OptimizationResult(
            evolution=evolution, rates=self._errors, best_solution=self._best_agent, task_type=task.minmax
        )

    def __error_check__(self) -> tuple[float, float, bool]:
        """
        Check whether the optimization algorithm has to stop or not based on the error criteria and the current cycle.
        :return: a tuple containing the current error, the average fitness and a boolean indicating whether the
        algorithm has to stop or not
        :rtype: tuple[float, float, bool]
        """
        # Get the optimal population
        avg_fit = average_fitness(self._population)
        current_error = abs(1 - avg_fit)
        previous_error = self._errors[-1] if len(self._errors) > 0 else 0

        # Append the current error to the list of errors
        self._errors.append(current_error)

        # Append the difference between the current error and the previous one to the list of error differences
        self._error_diffs.append(current_error - previous_error)

        return current_error, avg_fit, self.__should_stop__(current_error)

    def __should_stop__(self, current_error: float) -> bool:
        fitness_error = self._config.fitness_error
        max_cycles = self._config.max_cycles
        early_stopping = self._config.early_stopping

        cycle = self._current_cycle

        # Stop when the maximum number of cycles is reached
        has_to_stop = cycle >= max_cycles

        # Evaluate the early stopping criteria
        if early_stopping is not None:
            min_delta, patience = early_stopping.min_delta, early_stopping.patience
            has_to_stop |= all([diff < 0 and abs(diff) < min_delta for diff in self._error_diffs[-patience:]])

        # Stop when the error is below the error criteria
        if fitness_error is not None:
            has_to_stop |= current_error <= fitness_error

        return has_to_stop