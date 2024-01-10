import numpy as np

from ..helpers import (
    roulette_wheel_index,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Ant, AntColonyOptimizationConfig


class AntColonyOptimization(OptimizationAbstract):
    """
    Implementation of the Ant Colony Optimization algorithm.

    Args:
        config (AntColonyOptimizationConfig): an instance of AntColonyOptimizationConfig class.
            {parse_obj_doc(AntColonyOptimizationConfig)}

    Bibliography
    ----------
    [1] A. Colorni, M. Dorigo et V. Maniezzo, Distributed Optimization by Ant Colonies, actes de la première conférence
        européenne sur la vie artificielle, Paris, France, Elsevier Publishing, 134-142, 1991.
    [2] M. Dorigo, Optimization, Learning and Natural Algorithms, PhD thesis, Politecnico di Milano, Italy, 1992.
    """
    def __init__(self, config: AntColonyOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def compute_sigma(ant: Ant) -> float:
            M = np.repeat(np.array(ant.position).reshape((1, -1)), pop_size, axis=0)
            return zeta * np.sum(np.abs(positions - M), axis=0) / (pop_size - 1)

        def generate_coordinate(j: int) -> float:
            rdx = roulette_wheel_index(weights)
            return float(self._population[rdx].position[j] + np.random.normal() * sigmas[rdx, j])

        # compute the selection probability of each ant in the population
        ranks = np.array([idx for idx in range(1, self._config.population_size + 1)])
        Q = self._config.intent_factor * self._config.population_size
        weights = 1 / (np.sqrt(2 * np.pi) * Q) * np.exp(-0.5 * ((ranks - 1) / Q) ** 2)
        weights = weights / np.sum(weights)  # Normalize to find the probability

        # compute the sigmas of each ant in the population
        positions = np.array([ant.position for ant in self._population])
        pop_size = self._config.population_size
        zeta = self._config.zeta

        sigmas = np.array([compute_sigma(ant) for ant in self._population])

        # Generate new ants
        new_ants = [Ant(**self._init_agent(
            list(map(generate_coordinate, range(0, self._task.space_dimension)))
        ).model_dump()) for _ in range(0, self._config.archive_size)]

        self._extend_and_trim_population(new_ants)
