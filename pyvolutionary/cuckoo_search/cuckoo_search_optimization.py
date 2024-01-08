import numpy as np

from ..helpers import (
    get_levy_flight_step,
    sort_and_trim,
    parse_obj_doc  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Cuckoo, CuckooSearchOptimizationConfig


class CuckooSearchOptimization(OptimizationAbstract):
    """
    Implementation of the Cuckoo Search Optimization algorithm.

    Args:
        config (CuckooSearchOptimizationConfig): an instance of CuckooSearchOptimizationConfig class.
            {parse_obj_doc(CuckooSearchOptimizationConfig)}

    Bibliography
    ----------
    [1] Yang, X.S. and Deb, S., 2009, December. Cuckoo search via Lévy flights. In 2009 World congress on nature &
        biologically inspired computing (NaBIC) (pp. 210-214). Ieee.
    """
    def __init__(self, config: CuckooSearchOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__n_cut = int(self._config.p_a * self._config.population_size)

    def __evolve__(self, idx: int, cuckoo: Cuckoo, best_pos: list[float]) -> None:
        epoch = self._cycles
        pos = np.array(cuckoo.position)

        levy_step = get_levy_flight_step(multiplier=0.001, case=-1)
        new_agent = Cuckoo(**self._init_agent(
            pos + 1.0 / np.sqrt(epoch) * np.sign(np.random.random() - 0.5) * levy_step * (pos - best_pos)
        ).model_dump())

        return self._greedy_select_agent(new_agent, cuckoo)

    def optimization_step(self):
        self._population = self._solve_mode_process(self.__evolve__, self._best_agent.position)

        # abandoned some worst nests
        pop = sort_and_trim(self._population, self._config.population_size)
        self._population = (
            pop[:(self._config.population_size - self.__n_cut)] + [self._init_agent() for _ in range(0, self.__n_cut)]
        )