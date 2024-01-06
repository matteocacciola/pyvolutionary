import numpy as np

from ..helpers import (
    distance,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import PathfinderAlgorithmOptimizationConfig


class PathfinderAlgorithmOptimization(OptimizationAbstract):
    """
    Implementation of the Pathfinder Algorithm Optimization.

    Args:
        config (PathfinderAlgorithmOptimizationConfig): an instance of PathfinderAlgorithmOptimizationConfig class.
            {parse_obj_doc(PathfinderAlgorithmOptimizationConfig)}

    Bibliography
    ----------
    [1] Yapici, H. and Cetinkaya, N., 2019. A new meta-heuristic optimizer: Pathfinder algorithm. Applied soft
        computing, 78, pp.545-568.
    """
    def __init__(self, config: PathfinderAlgorithmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        alpha, beta = np.random.uniform(1, 2, 2)
        best_position = np.array(self._best_agent.position)

        A = self._uniform_position() * np.exp(-2 * self._cycles / self._config.max_cycles)
        t = 1. - self._cycles / (self._config.max_cycles + 1)
        space = self._bandwidth()
        
        # update the position of pathfinder and check the bound
        new_pathfinder = self._init_agent(np.array(self._population[0].position) + 2 * np.random.uniform() * (
            best_position - np.array(self._population[0].position)
        ) + A)

        # update positions of members, check the bound and calculate new fitness
        for idx in range(1, self._config.population_size):
            pathfinder_position = np.array(self._population[idx].position)
            pos_new = pathfinder_position.copy().astype(float)

            dist = distance(new_pathfinder.position, pathfinder_position.tolist()) / self._task.space_dimension
            pos_new += alpha * np.random.uniform() * (best_position - pathfinder_position) + (
                beta * np.random.uniform() * (self._population[idx - 1].position - pathfinder_position)
            ) + np.random.uniform() * t * (dist / space)

            pos_new = self._correct_position(pos_new)
            agent = self._init_agent(pos_new)

            self._population[idx] = self._greedy_select_agent(agent, self._population[idx])

        # update the position of the pathfinder
        self._population[0] = self._greedy_select_agent(new_pathfinder, self._population[0])
