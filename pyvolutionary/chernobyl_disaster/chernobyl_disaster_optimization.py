import numpy as np

from ..helpers import (
    best_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import SearchRadiation, ChernobylDisasterOptimizationConfig


class ChernobylDisasterOptimization(OptimizationAbstract):
    """
    Implementation of the Chernobyl Disaster Optimization algorithm.

    Args:
        config (ChernobylDisasterOptimizationConfig): an instance of ChernobylDisasterOptimizationConfig class.
            {parse_obj_doc(ChernobylDisasterOptimizationConfig)}

    Bibliography
    ----------
    [1] Shehadeh, H. A. (2023). Chernobyl disaster optimizer (CDO): a novel meta-heuristic method for global
        optimization. Neural Computing and Applications, 1-17.
    """
    def __init__(self, config: ChernobylDisasterOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __evolve__(
        self,
        idx: int,
        radiation: SearchRadiation,
        a1: float,
        a2: float,
        a3: float,
        b1_pos: list[float],
        b2_pos: list[float],
        b3_pos: list[float]
    ) -> SearchRadiation:
        a = 3. - 3. * self._cycles / self._config.max_cycles

        pos = np.array(radiation.position)
        b1_pos = np.array(b1_pos)
        b2_pos = np.array(b2_pos)
        b3_pos = np.array(b3_pos)

        pa = np.pi * np.random.random() ** 2 / (0.25 * a1) - a * np.random.random()
        pos_a = 0.25 * (b1_pos - pa * np.abs(np.random.random() ** 2 * np.pi * b1_pos - pos))

        pb = np.pi * np.random.random() ** 2 / (0.5 * a2) - a * np.random.random()
        pos_b = 0.5 * (b2_pos - pb * np.abs(np.random.random() ** 2 * np.pi * b2_pos - pos))

        pc = np.pi * np.random.random() ** 2 / a3 - a * np.random.random()
        pos_c = b3_pos - pc * np.abs(np.random.random() ** 2 * np.pi * b3_pos - pos)

        pos_new = (pos_a + pos_b + pos_c) / 3

        return SearchRadiation(**self._init_agent(pos_new).model_dump())

    def optimization_step(self):
        a1 = np.log10((16000 - 1) * np.random.random() + 16000)
        a2 = np.log10((270000 - 1) * np.random.random() + 270000)
        a3 = np.log10((300000 - 1) * np.random.random() + 300000)

        bests = best_agents(self._population, n_best=3)
        self._population = self._solve_mode_process(
            self.__evolve__, a1, a2, a3, bests[0].position, bests[1].position, bests[2].position
        )
