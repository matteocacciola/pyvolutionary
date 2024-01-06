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

    def __selection_probability__(self) -> np.ndarray:
        """
        Compute the selection probability of each ant in the population.
        :return: a numpy array of the selection probability of each ant in the population.
        :rtype: np.ndarray
        """
        ranks = np.array([idx for idx in range(1, self._config.population_size + 1)])
        Q = self._config.intent_factor * self._config.population_size
        weights = 1 / (np.sqrt(2 * np.pi) * Q) * np.exp(-0.5 * ((ranks - 1) / Q) ** 2)
        return weights / np.sum(weights)  # Normalize to find the probability.

    def __compute_sigmas__(self) -> np.ndarray:
        """
        Compute the sigmas of each ant in the population. Each sigma is computed as follows:
            sigma_i = zeta * sum(|x_j - x_i|) / (N - 1)
        where:
            x_i: the position of the i-th ant.\n
            x_j: the position of the j-th ant.\n
            zeta: the zeta parameter of the colony.\n
            N: the population size.
        :return: a numpy array of the sigmas of each ant in the population.
        :rtype: np.ndarray
        """
        positions = np.array([ant.position for ant in self._population])
        pop_size = self._config.population_size

        sigmas = []
        for idx in range(0, self._config.population_size):
            M = np.repeat(np.array(self._population[idx].position).reshape((1, -1)), pop_size, axis=0)
            sigmas.append(
                self._config.zeta * np.sum(np.abs(positions - M), axis=0) / (pop_size - 1)
            )
        return np.array(sigmas)

    def __generate_new_ants__(self, weights: np.ndarray, sigmas: np.ndarray) -> list[Ant]:
        def generate_coordinate(j: int) -> float:
            rdx = roulette_wheel_index(weights)
            return float(self._population[rdx].position[j] + np.random.normal() * sigmas[rdx, j])

        # Generate Samples
        return [Ant(**self._init_agent(
            list(map(generate_coordinate, range(0, self._task.space_dimension)))
        ).model_dump()) for _ in range(0, self._config.archive_size)]

    def optimization_step(self):
        weights = self.__selection_probability__()
        sigmas = self.__compute_sigmas__()

        # Generate new ants
        new_ants = self.__generate_new_ants__(weights, sigmas)

        self._extend_and_trim_population(new_ants)

