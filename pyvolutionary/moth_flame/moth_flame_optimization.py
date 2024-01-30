from typing import Any
import numpy as np

from ..helpers import (
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import MothFlameOptimizationConfig, MothFlame


class MothFlameOptimization(OptimizationAbstract):
    """
    Implementation of the Moth-Flame Optimization algorithm.

    Args:
        config (MothFlameOptimizationConfig): an instance of MothFlameOptimizationConfig class.
            {parse_obj_doc(MothFlameOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
        Knowledge-based systems, 89, pp.228-249.
    """
    def __init__(self, config: MothFlameOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = MothFlameOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(idx: int, moth_flame: MothFlame, moth_flame_sorted: MothFlame) -> MothFlame:
            pos = np.array(moth_flame.position)
            pos_sorted = np.array(moth_flame_sorted.position)
            # D in Eq.(3.13)
            distance_to_flame = np.abs(pos_sorted - pos)
            t = (a - 1) * np.random.uniform(0, 1, n_dims) + 1
            # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
            temp_1 = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pos_sorted
            # Update the position of the moth with respect to one flame Eq.(3.12).
            temp_2 = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + np.array(g_best.position)
            list_idx = idx * np.ones(n_dims)
            pos_new = np.where(list_idx < num_flame, temp_1, temp_2)
            agent = MothFlame(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(moth_flame, agent)

        cycle = self._current_cycle
        max_cycles = self._config.max_cycles
        n_dims = self._task.space_dimension
        pop_size = self._config.population_size

        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(pop_size - cycle * ((pop_size - 1) / max_cycles))
        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1. + cycle * (-1. / max_cycles)
        b = 1

        pop_sorted = sort_by_cost(self._population)
        g_best = self._best_agent

        self._population = [evolve(idx, moth_flame, pop_sorted[idx]) for idx, moth_flame in enumerate(self._population)]
