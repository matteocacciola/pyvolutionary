import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import SiberianTiger, SiberianTigerOptimizationConfig


class SiberianTigerOptimization(OptimizationAbstract):
    """
    Implementation of the Siberian Tiger Optimization algorithm.

    Args:
        config (SiberianTigerOptimizationConfig): an instance of SiberianTigerOptimizationConfig class.
            {parse_obj_doc(SiberianTigerOptimizationConfig)}

    Bibliography
    ----------
    [1] Trojovský, P., Dehghani, M., & Hanuš, P. (2022). Siberian Tiger Optimization: A New Bio-Inspired
        Metaheuristic Algorithm for Solving Engineering Optimization Problems. IEEE Access, 10, 132396-132431.
    """
    def __init__(self, config: SiberianTigerOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __get_indexes_better__(self, siberian_tiger: SiberianTiger) -> np.ndarray:
        costs = np.array([agent.cost for agent in self._population])
        idxs = np.where(costs < siberian_tiger.cost)
        return idxs[0]

    def __sf__(self, siberian_tiger: SiberianTiger) -> np.ndarray:
        idxs = self.__get_indexes_better__(siberian_tiger)
        if len(idxs) == 0 or np.random.random() < 0.5:
            return np.array(self._best_agent.position)

        kk = np.random.permutation(idxs)[0]
        return np.array(self._population[kk].position)

    def optimization_step(self):
        for idx, siberian_tiger in enumerate(self._population):
            pos = np.array(siberian_tiger.position)

            # phase 1: hunting (exploration)
            sf = self.__sf__(siberian_tiger)
            r1 = np.random.randint(1, 3)
            pos_new = pos + np.random.random() * (sf - r1 * pos)  # Eq. 5
            agent = self._init_agent(self._correct_position(pos_new))

            siberian_tiger = self._greedy_select_agent(siberian_tiger, agent)

            # phase 2: exploitation
            pos_new = self._increase_position(siberian_tiger.position, self._cycles)  # Eq. 7
            agent = self._init_agent(self._correct_position(pos_new))

            self._population[idx] = self._greedy_select_agent(siberian_tiger, agent)
