from typing import Any
import numpy as np

from ..helpers import (
    roulette_wheel_indexes,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Universe, MultiverseOptimizationConfig


class MultiverseOptimization(OptimizationAbstract):
    """
    Implementation of the Multi-verse Optimization algorithm.

    Args:
        config (MultiverseOptimizationConfig): an instance of MultiverseOptimizationConfig class.
            {parse_obj_doc(MultiverseOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., Mirjalili, S.M. and Hatamlou, A., 2016. Multi-verse optimizer: a nature-inspired
        algorithm for global optimization. Neural Computing and Applications, 27(2), pp.495-513.
    """
    def __init__(self, config: MultiverseOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = MultiverseOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(idx: int, universe: Universe) -> Universe:
            pos = np.array(universe.position)
            black_hole_pos = None
            if np.random.uniform() < wep:
                white_hole_id = roulette_wheel_indexes(list_costs)[0]
                black_hole_pos_1 = pos + tdr * np.random.normal(0, 1) * (
                    np.array(self._population[white_hole_id].position) - pos
                )
                black_hole_pos_2 = best_pos + tdr * np.random.normal(0, 1) * (best_pos - pos)
                black_hole_pos = np.where(np.random.random(n_dims) < 0.5, black_hole_pos_1, black_hole_pos_2)
            return self._greedy_select_agent(universe, Universe(**self._init_agent(black_hole_pos).model_dump()))

        epoch = self._current_cycle
        epochs = self._config.max_cycles
        wep_min, wep_max = self._config.wep_min, self._config.wep_max
        n_dims = self._task.space_dimension

        best_pos = np.array(self._best_agent.position)

        wep = wep_max - epoch * ((wep_max - wep_min) / epochs)
        tdr = 1 - epoch ** (1.0 / 6) / epochs ** (1.0 / 6)  # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        list_costs = np.array([universe.cost for universe in self._population])

        self._population = [evolve(idx, universe) for idx, universe in enumerate(self._population)]
