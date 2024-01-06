import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import SeagullOptimizationConfig


class SeagullOptimization(OptimizationAbstract):
    """
    Implementation of the Aquila Optimization algorithm.

    Args:
        config (SeagullOptimizationConfig): an instance of AquilaOptimizationConfig class.
            {parse_obj_doc(AquilaOptimizationConfig)}

    Bibliography
    ----------
    [1] Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: Theory and its applications for large-scale
        industrial engineering problems. Knowledge-based systems, 165, 169-196.
    """
    def __init__(self, config: SeagullOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        A = self._config.fc - self._cycles * self._config.fc / self._config.max_cycles  # Eq. 6
        uu = vv = 1

        best_pos = np.array(self._best_agent.position)
        for idx, seagull in enumerate(self._population):
            pos = np.array(seagull.position)

            B = 2 * A ** 2 * np.random.random()  # Eq. 8
            M = B * (best_pos - pos)  # Eq. 7
            C = A * pos  # Eq. 5
            D = np.abs(C + M)  # Eq. 9
            k = np.random.uniform(0, 2 * np.pi)
            r = uu * np.exp(k * vv)

            pos_new = r * np.cos(k) * r * np.sin(k) * r * k * D + np.random.normal(0, 1) * best_pos  # Eq. 14
            pos_new = self._correct_position(pos_new)
            agent = self._init_agent(pos_new)

            self._population[idx] = self._greedy_select_agent(agent, seagull)
