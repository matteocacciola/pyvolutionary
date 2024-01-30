from typing import Any
import numpy as np

from ..helpers import (
    best_agents,
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import GoldenJackalOptimizationConfig, GoldenJackal


class GoldenJackalOptimization(OptimizationAbstract):
    """
    Implementation of the Golden Jackal Optimization algorithm.

    Args:
        config (GoldenJackalOptimizationConfig): an instance of GoldenJackalOptimizationConfig class.
            {parse_obj_doc(GoldenJackalOptimizationConfig)}

    Bibliography
    ----------
    [1] Chopra, N., & Ansari, M. M. (2022). Golden jackal optimization: A novel nature-inspired optimizer for
        engineering applications. Expert Systems with Applications, 198, 116924.
    """
    def __init__(self, config: GoldenJackalOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = GoldenJackalOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(idx: int, jackal: GoldenJackal) -> GoldenJackal:
            pos = np.array(jackal.position)
            E = np.array(E1 * 2 * np.random.random(size=n_dims) - 1)
            male_position = np.array(male.position)
            female_position = np.array(female.position)
            t1 = np.abs(np.where(np.abs(E) < 1, RL[idx, :] * male_position - pos, male_position - RL[idx, :] * pos))
            t2 = np.abs(np.where(np.abs(E) < 1, RL[idx, :] * female_position - pos, female_position - RL[idx, :] * pos))
            male_position -= E * t1
            female_position -= E * t2
            pos_new = ((male_position + female_position) / 2).tolist()
            return self._greedy_select_agent(jackal, GoldenJackal(**self._init_agent(pos_new).model_dump()))

        n_dims = self._task.space_dimension
        pop_size = self._config.population_size

        E1 = 1.5 * (1. - (self._current_cycle / self._config.max_cycles))
        RL = get_levy_flight_step(beta=1.5, multiplier=0.05, size=(pop_size, n_dims), case=-1)
        male, female = best_agents(self._population, n_best=2)

        self._population = [evolve(idx, jackal) for idx, jackal in enumerate(self._population)]
