from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import CoatiOptimizationConfig, Coati


class CoatiOptimization(OptimizationAbstract):
    """
    Implementation of the Coati Optimization algorithm.

    Args:
        config (CoatiOptimizationConfig): an instance of CoatiOptimizationConfig class.
            {parse_obj_doc(CoatiOptimizationConfig)}

    Bibliography
    ----------
    [1] Dehghani, M., Montazeri, Z., Trojovská, E., & Trojovský, P. (2023). Coati Optimization Algorithm: A new
        bio-inspired metaheuristic algorithm for solving optimization problems. Knowledge-Based Systems, 259, 110011.
    """
    def __init__(self, config: CoatiOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__size2: int | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = CoatiOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__size2 = int(self._config.population_size / 2)

    def optimization_step(self):
        def hunting(coati: Coati) -> Coati:
            pos = np.array(coati.position)
            pos_new = pos + np.random.random() * (best_pos - np.random.randint(1, 3) * pos)  # Eq. 4
            return self._greedy_select_agent(coati, Coati(**self._init_agent(pos_new).model_dump()))

        def attacking(coati: Coati) -> Coati:
            pos = np.array(coati.position)
            iguana = self._init_agent()
            if iguana.cost < coati.cost:
                pos_new = pos + np.random.random() * (
                    np.array(iguana.position) - np.random.randint(1, 3) * pos
                )  # Eq. 6
                return self._greedy_select_agent(coati, Coati(**self._init_agent(pos_new).model_dump()))
            pos_new = pos + np.random.random() * (pos - np.array(iguana.position))  # Eq. 6
            return self._greedy_select_agent(coati, Coati(**self._init_agent(pos_new).model_dump()))

        def exploration(idx: int, coati: Coati) -> Coati:
            if idx < self.__size2:
                return hunting(coati)
            return attacking(coati)

        def exploitation(coati: Coati) -> Coati:
            pos = np.array(coati.position)
            pos_new = pos + (1 - 2 * np.random.random()) * (low + np.random.random() * (high - low))  # Eq. 8
            return self._greedy_select_agent(coati, Coati(**self._init_agent(pos_new).model_dump()))

        best_pos = np.array(self._best_agent.position)

        # phase1: hunting and attacking strategy on iguana (exploration phase)
        self._population = [exploration(idx, coati) for idx, coati in enumerate(self._population)]

        # phase2: the process of escaping from predators (exploitation phase)
        lb, ub = self._task.get_bounds()
        low, high = lb / self._current_cycle, ub / self._current_cycle
        self._population = [exploitation(coati) for coati in self._population]
