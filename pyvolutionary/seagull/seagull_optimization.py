import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import SeagullOptimizationConfig, Seagull


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

    def __evolve__(self, idx: int, seagull: Seagull, best_position: list[float]) -> Seagull:
        best_position = np.array(best_position)

        A = self._config.fc - self._cycles * self._config.fc / self._config.max_cycles  # Eq. 6
        uu = vv = 1

        pos = np.array(seagull.position)

        B = 2 * A ** 2 * np.random.random()  # Eq. 8
        M = B * (best_position - pos)  # Eq. 7
        C = A * pos  # Eq. 5
        D = np.abs(C + M)  # Eq. 9
        k = np.random.uniform(0, 2 * np.pi)
        r = uu * np.exp(k * vv)

        pos_new = r * np.cos(k) * r * np.sin(k) * r * k * D + np.random.normal(0, 1) * best_position  # Eq. 14
        agent = Seagull(**self._init_agent(pos_new).model_dump())

        return self._greedy_select_agent(agent, seagull)

    def optimization_step(self):
        self._population = self._solve_mode_process(self.__evolve__, self._best_agent.position)
