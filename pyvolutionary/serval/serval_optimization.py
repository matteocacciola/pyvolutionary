from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import ServalOptimizationConfig, Serval


class ServalOptimization(OptimizationAbstract):
    """
    Implementation of the Serval Optimization algorithm.

    Args:
        config (ServalOptimizationConfig): an instance of ServalOptimizationConfig class.
            {parse_obj_doc(ServalOptimizationConfig)}

    Bibliography
    ----------
    [1] Dehghani, M., & Trojovský, P. (2022). Serval Optimization Algorithm: A New Bio-Inspired Approach for Solving
        Optimization Problems. Biomimetics, 7(4), 204.
    """
    def __init__(self, config: ServalOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = ServalOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(serval: Serval) -> Serval:
            # phase 1: Prey Selection and Attacking (Exploration)
            pos = np.array(serval.position)
            pos_new = pos + np.random.random(n_dims) * (kk_pos - np.random.randint(1, 3, n_dims) * pos)
            agent = self._greedy_select_agent(serval, Serval(**self._init_agent(pos_new).model_dump()))
            # phase 2: Chase Process (Exploitation)
            pos_new = pos + np.random.randint(1, 3, n_dims) * bandwidth / epoch  # Eq. 6
            return self._greedy_select_agent(agent, Serval(**self._init_agent(pos_new).model_dump()))

        kk_pos = np.array(self._population[np.random.permutation(self._config.population_size)[0]].position)
        n_dims = self._task.space_dimension
        bandwidth = self._task.bandwidth()
        epoch = self._current_cycle

        self._population = [evolve(serval) for serval in self._population]
