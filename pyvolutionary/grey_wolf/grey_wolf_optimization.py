import numpy as np

from ..helpers import (
    best_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import GreyWolf, GreyWolfOptimizationConfig


class GreyWolfOptimization(OptimizationAbstract):
    """
    Implementation of the Grey Wolf Optimization algorithm.

    Args:
        config (GreyWolfOptimizationConfig): an instance of GreyWolfOptimizationConfig class.
            {parse_obj_doc(GreyWolfOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69,
        46-61. Chicago
    """
    def __init__(self, config: GreyWolfOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__alpha_wolf: GreyWolf | None = None
        self.__beta_wolf: GreyWolf | None = None
        self.__gamma_wolf: GreyWolf | None = None

    def _init_population(self):
        super()._init_population()
        self.__alpha_wolf, self.__beta_wolf, self.__gamma_wolf = best_agents(self._population, 3)

    def optimization_step(self):
        """
        Implementation of the Grey Wolf Optimization algorithm. [1]
        """
        # linearly decreased from 2 to 0
        a = 2 * (1 - self._cycles / self._config.max_cycles)

        # updating each population member with the help of best three members
        for idx in range(0, self._config.population_size):
            position = np.array(self._population[idx].position)
            a1, a2, a3 = (a * (2 * np.random.random() - 1),
                          a * (2 * np.random.random() - 1),
                          a * (2 * np.random.random() - 1))
            c1, c2, c3 = 2 * np.random.random(), 2 * np.random.random(), 2 * np.random.random()

            x1 = np.array(self.__alpha_wolf.position) - a1 * np.abs(
                c1 * np.array(self.__alpha_wolf.position) - position
            )
            x2 = np.array(self.__beta_wolf.position) - a2 * np.abs(c2 * np.array(self.__beta_wolf.position) - position)
            x3 = np.array(self.__gamma_wolf.position) - a3 * np.abs(
                c3 * np.array(self.__gamma_wolf.position) - position
            )
            # greedy selection
            self._population[idx] = self._greedy_select_agent(
                self._population[idx], self._init_agent((x1 + x2 + x3) / 3)
            )

        # best 3 solutions will be called as alpha, beta and gaama
        self.__alpha_wolf, self.__beta_wolf, self.__gamma_wolf = best_agents(self._population, 3)
