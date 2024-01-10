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
    def __init__(self, config: MonarchButterflyOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__bar = self._config.partition
        self.__np1 = int(np.ceil(self._config.partition * self._config.population_size))
        self.__np2 = self._config.population_size - self.__np1

    def optimization_step(self):
        # get the elite agents of the current generation
        sort_by_cost(self._population)
        elite = self._population[:self._config.keep].copy()

        # apply the migration operator to the best habitats of the current generation
        r1 = np.random.random(size=self.__np1) * self._config.period
        partition_condition = r1 <= self._config.partition
        n_values = np.where(partition_condition, self.__np1, self.__np2)

        pop_indices = np.round(n_values * np.random.random(size=self.__np1) + 0.5).astype(int).tolist()

        pop1 = [
            MonarchButterfly(**self._init_agent(self._population[idx].position).model_dump()) for idx in pop_indices
        ]

        # apply the adjusting operator to the worst habitats of the current generation
        scale = 1.0 / ((self._current_cycle + 1) ** 2)
        step_size = np.ceil(np.random.exponential(2 * self._config.max_cycles))
        delta_x = get_levy_flight_step(beta=1., multiplier=step_size, size=self._task.space_dimension, case=1)

        mask = np.random.uniform(0.0, 1.0, size=self.__np2) <= self._config.partition
        indices = np.where(
            mask, -1, np.round(self.__np2 * np.random.random(size=self.__np2) + 0.5).astype(int).tolist()
        )

        positions = [self._best_agent.position if idx == -1 else (
            self._population[idx].position + scale * (delta_x - 0.5) * int(np.random.uniform(0.0, 1.0) > self.__bar)
        ) for idx in indices]
        pop2 = [MonarchButterfly(**self._init_agent(position).model_dump()) for position in positions]

        # apply the elitism operator
        self._population = sort_and_trim(pop1 + pop2, self._config.population_size - self._config.keep)
        self._population.extend(elite)
