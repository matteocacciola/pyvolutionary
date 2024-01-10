import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Fox, FoxOptimizationConfig


class FoxOptimization(OptimizationAbstract):
    """
    Implementation of the Fox Optimization algorithm.

    Args:
        config (FoxOptimizationConfig): an instance of FoxOptimizationConfig class.
            {parse_obj_doc(FoxOptimizationConfig)}

    Bibliography
    ----------
    [1] Mohammed, H., & Rashid, T. (2023). FOX: a FOX-inspired optimization algorithm. Applied Intelligence, 53(1),
        1030-1050.
    """
    def __init__(self, config: FoxOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__mint = np.inf

    def optimization_step(self):
        def evolve(fox: Fox) -> Fox:
            if np.random.random() >= 0.5:
                time1 = np.random.random(dim)
                sps = best_position / time1
                travel_distance = 0.5 * sps * time1
                tt = np.mean(time1)
                self.__mint = min(self.__mint, tt)
                jump = 0.5 * 9.81 * (tt / 2) ** 2
                pos_new = travel_distance * jump * (c1 if np.random.random() > 0.18 else c2)
            else:
                pos_new = best_position + np.random.standard_normal(dim) * (self.__mint * a)
            agent = Fox(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(fox, agent)

        a = 2 * (1 - (1.0 / self._current_cycle))
        c1 = self._config.c1
        c2 = self._config.c2
        dim = self._task.space_dimension
        best_position = np.array(self._best_agent.position)

        self._population = [evolve(fox) for fox in self._population]
