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

    def optimization_step(self):
        def get_indexes_better(siberian_tiger: SiberianTiger) -> np.ndarray:
            idxs = np.where(costs < siberian_tiger.cost)
            return idxs[0]

        def sf(siberian_tiger: SiberianTiger) -> np.ndarray:
            idxs = get_indexes_better(siberian_tiger)
            if len(idxs) == 0 or np.random.random() < 0.5:
                return beet_position
            kk = np.random.permutation(idxs)[0]
            return np.array(self._population[kk].position)

        def evolve(siberian_tiger: SiberianTiger) -> SiberianTiger:
            pos = np.array(siberian_tiger.position)
            # phase 1: hunting (exploration)
            sf_res = sf(siberian_tiger)
            r1 = np.random.randint(1, 3)
            pos_new = pos + np.random.random() * (sf_res - r1 * pos)  # Eq. 5
            agent = SiberianTiger(**self._init_agent(pos_new).model_dump())
            siberian_tiger = self._greedy_select_agent(siberian_tiger, agent)
            # phase 2: exploitation
            pos_new = self._increase_position(siberian_tiger.position, current_cycle)  # Eq. 7
            agent = SiberianTiger(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(siberian_tiger, agent)

        current_cycle = self._current_cycle
        costs = np.array([agent.cost for agent in self._population])
        beet_position = np.array(self._best_agent.position)

        self._population = [evolve(siberian_tiger) for siberian_tiger in self._population]
