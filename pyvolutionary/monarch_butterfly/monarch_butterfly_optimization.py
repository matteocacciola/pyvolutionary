from typing import Any
import numpy as np

from ..helpers import (
    get_levy_flight_step,
    sort_and_trim,
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import MonarchButterfly, MonarchButterflyOptimizationConfig


class MonarchButterflyOptimization(OptimizationAbstract):
    """
    Implementation of the Monarch Butterfly Optimization algorithm.

    Args:
        config (MonarchButterflyOptimizationConfig): an instance of MonarchButterflyOptimizationConfig class.
            {parse_obj_doc(MonarchButterflyOptimizationConfig)}

    Bibliography
    ----------
    [1] Wang, G. G., Deb, S., & Cui, Z. (2019). Monarch butterfly optimization. Neural computing and applications,
        31(7), 1995-2014.
    """
    def __init__(self, config: MonarchButterflyOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__bar: float | None = None
        self.__np1: int | None = None
        self.__np2: int | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = MonarchButterflyOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__bar = self._config.partition
        # both lands hold at least one butterfly
        self.__np1 = int(min(max(np.ceil(self._config.partition * self._config.population_size), 1),
                             self._config.population_size - 1))
        self.__np2 = self._config.population_size - self.__np1

    def optimization_step(self):
        # get the elite agents of the current generation
        self._population = sort_by_cost(self._population)
        elite = self._population[:self._config.keep].copy()

        n_dims = self._task.space_dimension
        dims = np.arange(n_dims)
        partition, period = self._config.partition, self._config.period
        # Land 1 holds the best butterflies, Land 2 the other ones
        land1 = np.array([butterfly.position for butterfly in self._population[:self.__np1]], dtype=float)
        land2 = np.array([butterfly.position for butterfly in self._population[self.__np1:]], dtype=float)

        # migration operator (Land 1): each dimension comes from a random butterfly of Land 1 or of Land 2
        pop1 = []
        for _ in range(0, self.__np1):
            from_land1 = np.random.random(n_dims) * period <= partition
            position = np.where(
                from_land1,
                land1[np.random.randint(0, self.__np1, n_dims), dims],
                land2[np.random.randint(0, self.__np2, n_dims), dims],
            )
            pop1.append(MonarchButterfly(**self._init_agent(position).__dict__))

        # butterfly adjusting operator (Land 2): each dimension comes from the best butterfly or from a random one of
        # Land 2, which may take a Levy flight step
        best_position = np.array(self._best_agent.position)
        alpha = 1.0 / (self._current_cycle ** 2)
        pop2 = []
        for _ in range(0, self.__np2):
            step_size = np.ceil(np.random.exponential(2 * self._config.max_cycles))
            delta_x = get_levy_flight_step(beta=1., multiplier=step_size, size=n_dims, case=1)
            from_best = np.random.random(n_dims) <= partition
            position = land2[np.random.randint(0, self.__np2, n_dims), dims]
            flies = np.random.random(n_dims) > self.__bar
            position = np.where(flies, position + alpha * (delta_x - 0.5), position)
            position = np.where(from_best, best_position, position)
            pop2.append(MonarchButterfly(**self._init_agent(position).__dict__))

        # apply the elitism operator
        self._population = sort_and_trim(pop1 + pop2, self._config.population_size - self._config.keep)
        self._population.extend(elite)
