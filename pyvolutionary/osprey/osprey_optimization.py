import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Osprey, OspreyOptimizationConfig


class OspreyOptimization(OptimizationAbstract):
    """
    Implementation of the Osprey Optimization algorithm.

    Args:
        config (OspreyOptimizationConfig): an instance of OspreyOptimizationConfig class.
            {parse_obj_doc(OspreyOptimizationConfig)}

    Bibliography
    ----------
    [1] Trojovský, P., & Dehghani, M. Osprey Optimization Algorithm: A new bio-inspired metaheuristic algorithm
        for solving engineering optimization problems. Frontiers in Mechanical Engineering, 8, 136.
    """
    def __init__(self, config: OspreyOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__phi = (1 + np.sqrt(5)) / 2  # golden ratio

    def __get_indexes_better__(self, osprey: Osprey) -> np.ndarray:
        costs = np.array([agent.cost for agent in self._population])
        idxs = np.where(costs < osprey.cost)
        return idxs[0]

    def __sf__(self, osprey: Osprey) -> np.ndarray:
        idxs = self.__get_indexes_better__(osprey)
        if len(idxs) == 0 or np.random.random() < 0.5:
            return np.array(self._best_agent.position)

        kk = np.random.permutation(idxs)[0]
        return np.array(self._population[kk].position)

    def optimization_step(self):
        for idx, osprey in enumerate(self._population):
            pos = np.array(osprey.position)

            # phase 1: position identification and fish hunting (exploration)
            sf = self.__sf__(osprey)
            r1 = np.random.randint(1, 3)
            pos_new = pos + np.random.normal(0, 1) * (sf - r1 * pos)  # Eq. 5
            agent = self._init_agent(self._correct_position(pos_new))

            osprey = self._greedy_select_agent(osprey, agent)

            # phase 2: carrying the fish to a suitable position (exploitation)
            pos_new = self._increase_position(osprey.position)  # Eq. 7
            agent = self._init_agent(self._correct_position(pos_new))

            self._population[idx] = self._greedy_select_agent(osprey, agent)
