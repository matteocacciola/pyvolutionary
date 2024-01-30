from typing import Any
import numpy as np

from ..helpers import (
    roulette_wheel_indexes,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import GerminalCenter, GerminalCenterOptimizationConfig


class GerminalCenterOptimization(OptimizationAbstract):
    """
    Implementation of the Germinal Center Optimization algorithm.

    Args:
        config (GerminalCenterOptimizationConfig): an instance of GerminalCenterOptimizationConfig class.
            {parse_obj_doc(GerminalCenterOptimizationConfig)}

    Bibliography
    ----------
    [1] Villaseñor, C., Arana-Daniel, N., Alanis, A.Y., López-Franco, C. and Hernandez-Vargas, E.A., 2018.
    Germinal center optimization algorithm. International Journal of Computational Intelligence Systems, 12(1), p.13.
    """
    def __init__(self, config: GerminalCenterOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = GerminalCenterOptimizationConfig(**parameters)

    def _init_agent(
        self, position: list[Any] | np.ndarray | None = None, cc: float | None = None, ls: float | None = None
    ) -> GerminalCenter:
        agent = super()._init_agent(position=position)
        return GerminalCenter(**agent.model_dump(), cell_counter=cc, life_signal=ls)

    def optimization_step(self):
        def dark_zone(center: GerminalCenter) -> GerminalCenter:
            pos = np.array(center.position)
            if np.random.uniform(0, 100) < center.life_signal:
                center.cell_counter += 1
            elif center.cell_counter > 1:
                center.cell_counter -= 1
            counters = np.array([center.cell_counter for center in self._population])
            r1, r2, r3 = roulette_wheel_indexes(counters, 3)
            pos_new = np.array(self._population[r1].position) + wf * (
                np.array(self._population[r2].position) - np.array(self._population[r3].position)
            )
            pos_new = np.where(np.random.random(n_dims) < cr, pos_new, pos)
            agent = self._init_agent(pos_new, center.cell_counter, center.life_signal + 10)
            return self._greedy_select_agent(center, agent)

        def light_zone(center: GerminalCenter) -> GerminalCenter:
            center.life_signal -= 10
            cost = (center.cost - cost_max) / (cost_min - cost_max)
            center.life_signal += 10 * cost
            return center

        cr = self._config.cr
        wf = self._config.wf
        n_dims = self._task.space_dimension

        # Dark zone process
        self._population = [dark_zone(agent) for agent in self._population]

        # Light zone process
        costs_list = np.array([agent.cost for agent in self._population])
        cost_max = np.max(costs_list)
        cost_min = np.min(costs_list)

        self._population = [light_zone(agent) for agent in self._population]
