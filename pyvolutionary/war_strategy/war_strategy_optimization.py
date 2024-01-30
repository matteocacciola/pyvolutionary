from typing import Any
import numpy as np

from ..helpers import (
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Soldier, WarStrategyOptimizationConfig


class WarStrategyOptimization(OptimizationAbstract):
    """
    Implementation of the War Strategy Optimization algorithm.

    Args:
        config (WarStrategyOptimizationConfig): an instance of WarStrategyOptimizationConfig class.
            {parse_obj_doc(WarStrategyOptimizationConfig)}

    Bibliography
    ----------
    [1] Ayyarao, Tummala SLV, and Polamarasetty P. Kumar. "Parameter estimation of solar PV models with a new proposed
        war strategy optimization algorithm." International Journal of Energy Research (2022).
    """
    def __init__(self, config: WarStrategyOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = WarStrategyOptimizationConfig(**parameters)

    def _init_agent(
        self, position: list[Any] | np.ndarray | None = None, wl: float | None = None, wg: float | None = None
    ) -> Soldier:
        agent = super()._init_agent(position=position)
        return Soldier(**agent.model_dump(), wl=wl, wg=wg)

    def optimization_step(self):
        def evolve(idx: int, soldier: Soldier) -> Soldier:
            pos = np.array(soldier.position)
            r1 = np.random.random()
            if r1 < rr:
                pos_new = 2 * r1 * (best_position - np.array(self._population[com[idx]].position)) + (
                    soldier.wl * np.random.random() * (np.array(pop_sorted[idx].position) - pos)
                )
            else:
                pos_new = 2 * r1 * (np.array(pop_sorted[idx].position) - best_position) + (
                    np.random.random() * (soldier.wl * best_position - pos)
                )
            agent = self._init_agent(pos_new, soldier.wg + 1, soldier.wl * (1 - soldier.wg / epochs) ** 2)
            return self._greedy_select_agent(soldier, agent)

        rr = self._config.rr
        pop_sorted = sort_by_cost(self._population)
        com = np.random.permutation(self._config.population_size)

        best_position = np.array(self._best_agent.position)
        epochs = self._config.max_cycles

        self._population = [evolve(idx, soldier) for idx, soldier in enumerate(self._population)]
