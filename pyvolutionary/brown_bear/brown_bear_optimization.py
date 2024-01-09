from typing import Final
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import BrownBear, BrownBearOptimizationConfig


class BrownBearOptimization(OptimizationAbstract):
    """
    Implementation of the Brown-Bear Optimization algorithm.

    Args:
        config (BrownBearOptimizationConfig): an instance of BrownBearOptimizationConfig class.
            {parse_obj_doc(BrownBearOptimizationConfig)}

    Bibliography
    ----------
    [1] Prakash, T., Singh, P. P., Singh, V. P., & Singh, S. N. (2023). A Novel Brown-bear Optimization Algorithm for
        Solving Economic Dispatch Problem. In Advanced Control & Optimization Paradigms for Energy System Operation and
        Management (pp. 137-164). River Publishers.
    """
    EPS: Final[float] = np.finfo(float).eps
    
    def __init__(self, config: BrownBearOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __pedal_marking__(
        self, idx: int, bear: BrownBear, pp: float, best_position: list[float], worst_position: list[float]
    ) -> BrownBear:
        pos = np.array(bear.position)
        best_position = np.array(best_position)
        worst_position = np.array(worst_position)
        n_dims = self._task.space_dimension

        if pp <= self._cycles / 3:  # gait while walking
            pos_new = pos + (-pp * np.random.random(n_dims) * pos)
        elif self._cycles / 3 < pp <= 2 * self._cycles / 3:  # Careful Stepping
            qq = pp * np.random.random(n_dims)
            pos_new = pos + (qq * (best_position - np.random.randint(1, 3) * worst_position))
        else:
            ww = 2 * pp * np.pi * np.random.random(n_dims)
            pos_new = pos + (ww * best_position - np.abs(pos)) - (ww * worst_position - np.abs(pos))

        return self._greedy_select_agent(bear, BrownBear(**self._init_agent(pos_new).model_dump()))

    def __sniffing_bear__(self, idx: int, bear: BrownBear) -> BrownBear:
        pos = np.array(bear.position)
        kk = np.random.choice(list(set(range(0, self._config.population_size)) - {idx}))
        agent = self._population[kk]
        agent_pos = np.array(agent.position)
        pos_new = pos + np.random.random() * (pos - agent_pos) * (1 if bear.cost < agent.cost else -1)
        return self._greedy_select_agent(bear, BrownBear(**self._init_agent(pos_new).model_dump()))

    def optimization_step(self):
        pp = self._cycles / self._config.max_cycles

        self._population = self._solve_mode_process(
            self.__pedal_marking__, pp, self._best_agent.position, self._worst_agent.position,
        )

        self._population = [self.__sniffing_bear__(idx, bear) for idx, bear in enumerate(self._population)]
