from typing import Any
import numpy as np

from ..helpers import (
    best_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import SuccessHistoryIntelligentOptimizationConfig, Solution


class SuccessHistoryIntelligentOptimization(OptimizationAbstract):
    """
    Implementation of the Success History Intelligent Optimization algorithm.

    Args:
        config (SuccessHistoryIntelligentOptimizationConfig): an instance of SuccessHistoryIntelligentOptimizationConfig
            class.
            {parse_obj_doc(SuccessHistoryIntelligentOptimizationConfig)}

    Bibliography
    ----------
    [1] Fakhouri, H. N., Hamad, F., & Alawamrah, A. (2022). Success history intelligent optimizer. The Journal of
        Supercomputing, 1-42.
    """
    def __init__(self, config: SuccessHistoryIntelligentOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__a = 1.5

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = SuccessHistoryIntelligentOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(solution: Solution) -> Solution:
            pos = np.array(solution.position)
            x1 = b1_pos + (a * 2 * np.random.random(n_dims) - a) * np.abs(np.random.random(n_dims) * b1_pos - pos)
            x2 = b2_pos + (a * 2 * np.random.random(n_dims) - a) * np.abs(np.random.random(n_dims) * b2_pos - pos)
            x3 = b3_pos + (a * 2 * np.random.random(n_dims) - a) * np.abs(np.random.random(n_dims) * b3_pos - pos)
            pos_new = (x1 + x2 + x3) / 3
            return self._greedy_select_agent(solution, Solution(**self._init_agent(pos_new).model_dump()))
        
        b1_pos, b2_pos, b3_pos = map(lambda x: np.array(x.position), best_agents(self._population, n_best=3))
        self.__a -= 0.04
        a = self.__a

        n_dims = self._task.space_dimension

        self._population = [evolve(solution) for solution in self._population]
