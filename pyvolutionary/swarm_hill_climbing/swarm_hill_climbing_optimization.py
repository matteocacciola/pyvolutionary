from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import SwarmHillClimbingOptimizationConfig, Climber


class SwarmHillClimbingOptimization(OptimizationAbstract):
    """
    Implementation of the Swarm Hill Climbing Optimization algorithm.

    Args:
        config (SwarmHillClimbingOptimizationConfig): an instance of SwarmHillClimbingOptimizationConfig class.
            {parse_obj_doc(SwarmHillClimbingOptimizationConfig)}

    Bibliography
    ----------
    [1] Mitchell, M., Holland, J. and Forrest, S., 1993. When will a genetic algorithm outperform hill climbing.
        Advances in neural information processing systems, 6.
    """
    def __init__(self, config: SwarmHillClimbingOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = SwarmHillClimbingOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(idx: int, climber: Climber) -> Climber:
            pos = np.array(climber.position)
            c_ss = ss[idx]
            best_local = best_agent([Climber(
                **self._init_agent(pos + np.random.normal(0, 1, n_dims) * c_ss).model_dump()
            ) for _ in range(0, neighbour_size)])
            return self._greedy_select_agent(climber, best_local)
        
        pop_size = self._config.population_size
        epoch = self._current_cycle
        epochs = self._config.max_cycles
        n_dims = self._task.space_dimension

        neighbour_size = self._config.neighbour_size

        ranks = np.array(list(range(1, pop_size + 1)))
        ss = np.mean(self._task.bandwidth()) * np.exp(-2 * (epoch + 1) / epochs) * ranks / np.sum(ranks)

        self._population = [evolve(idx, climber) for idx, climber in enumerate(self._population)]
