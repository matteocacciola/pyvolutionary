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

    def __evolve__(self, idx: int, fox: Fox, best_position: list[float]) -> Fox:
        a = 2 * (1 - (1.0 / self._cycles))
        best_position = np.array(best_position)

        if np.random.random() >= 0.5:
            time1 = np.random.random(self._task.space_dimension)
            sps = best_position / time1
            travel_distance = 0.5 * sps * time1
            tt = np.mean(time1)
            jump = 0.5 * 9.81 * (tt / 2) ** 2
            pos_new = travel_distance * jump * (self._config.c1 if np.random.random() > 0.18 else self._config.c2)
            if self.__mint > tt:
                self.__mint = tt
        else:
            pos_new = best_position + np.random.standard_normal(self._task.space_dimension) * (self.__mint * a)

        agent = Fox(**self._init_agent(pos_new).model_dump())
        return self._greedy_select_agent(fox, agent)

    def optimization_step(self):
        self._population = self._solve_mode_process(self.__evolve__, self._best_agent.position)
