import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import WalrusOptimizationConfig


class WalrusOptimization(OptimizationAbstract):
    """
    Implementation of the Walrus Optimization algorithm.

    Args:
        config (WalrusOptimizationConfig): an instance of WalrusOptimizationConfig class.
            {parse_obj_doc(OspreyOptimizationConfig)}

    Bibliography
    ----------
    [1] Trojovský, P., & Dehghani, M. (2022). Walrus Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm.
    """
    def __init__(self, config: WalrusOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        pop_size = self._config.population_size

        for idx, walrus in enumerate(self._population):
            # Phase 1: Feeding strategy (exploration), with Eq. 4
            kk = np.random.permutation(pop_size)[0]
            agent_kk = self._population[kk]

            pos = np.array(walrus.position)
            pos_new = pos + np.random.random() * (
                np.array(agent_kk.position) - np.random.randint(1, 3) * pos
            ) if agent_kk.cost < walrus.cost else pos + np.random.random() * (pos - np.array(agent_kk.position))

            agent = self._init_agent(self._correct_position(pos_new))
            walrus = self._greedy_select_agent(walrus, agent)

            # phase 2 Exploitation
            pos_new = self._increase_position(walrus.position, self._cycles)  # Eq. 7
            agent = self._init_agent(self._correct_position(pos_new))

            self._population[idx] = self._greedy_select_agent(walrus, agent)
