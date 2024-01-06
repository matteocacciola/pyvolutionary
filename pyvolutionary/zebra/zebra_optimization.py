import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import ZebraOptimizationConfig


class ZebraOptimization(OptimizationAbstract):
    """
    Implementation of the Zebra Optimization algorithm.

    Args:
        config (ZebraOptimizationConfig): an instance of ZebraOptimizationConfig class.
            {parse_obj_doc(ZebraOptimizationConfig)}

    Bibliography
    ----------
    [1] Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Zebra optimization algorithm: A new bio-inspired
        optimization algorithm for solving optimization algorithm. IEEE Access, 10, 49445-49473.
    """
    def __init__(self, config: ZebraOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        n_dims = self._task.space_dimension
        
        best_pos = np.array(self._best_agent.position)

        # phase 1: foraging behaviour
        for idx, zebra in enumerate(self._population):
            pos = np.array(zebra.position)
            
            r1 = np.round(1 + np.random.random())
            pos_new = pos + np.random.random(n_dims) * (best_pos - r1 * pos)  # Eq. 3
            agent = self._init_agent(self._correct_position(pos_new))
            self._population[idx] = self._greedy_select_agent(agent, zebra)

        # phase 2: defense strategies against predators
        kk = np.random.permutation(self._config.population_size)[0]
        pos_kk = np.array(self._population[kk].position)
        for idx, zebra in enumerate(self._population):
            pos = np.array(zebra.position)

            # strategy 1: the lion attacks the zebra and thus the zebra chooses an escape strategy OR
            # strategy 2: other predators attack the zebra and the zebra will choose the offensive strategy
            pos_new = pos + 0.1 * (2 + np.random.random(n_dims) - 1) * (
                1 - self._cycles / self._config.max_cycles
            ) * pos if np.random.random() < 0.5 else pos + np.random.random(n_dims) * (
                pos_kk - np.random.randint(1, 3) * pos
            )

            agent = self._init_agent(self._correct_position(pos_new))
            self._population[idx] = self._greedy_select_agent(agent, zebra)
