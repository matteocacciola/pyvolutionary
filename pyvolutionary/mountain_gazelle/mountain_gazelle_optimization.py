import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import MountainGazelleOptimizationConfig


class MountainGazelleOptimization(OptimizationAbstract):
    """
    Implementation of the Mountain Gazelle Optimization algorithm.

    Args:
        config (MountainGazelleOptimizationConfig): an instance of MountainGazelleSwarmOptimizationConfig class.
            {parse_obj_doc(MountainGazelleOptimizationConfig)}

    Bibliography
    ----------
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., Khodadadi, N., & Mirjalili, S. (2022). Mountain gazelle optimizer: a
        new nature-inspired metaheuristic algorithm for global optimization problems. Advances in Engineering Software,
        174, 103282.
    """
    def __init__(self, config: MountainGazelleOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __random_solution__(self) -> np.ndarray:
        """
        Random solution. It is used to generate a new solution. It is a random combination of the best solution and the
        mean of the positions of the best solutions.
        :return: random solution.
        :rtype: np.ndarray
        """
        pop_size = self._config.population_size
        mean_positions = np.average(
            np.array([
                self._population[mm].position for mm in np.random.permutation(pop_size)[:int(np.ceil(pop_size / 3))]
            ]),
            axis=0,
        )

        index = np.random.randint(int(np.ceil(pop_size / 3)), pop_size)
        return (
            np.array(self._population[index].position) * np.floor(np.random.normal()) +
            mean_positions * np.ceil(np.random.normal())
        )

    def __coefficient_vectors__(self, epoch: int, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the coefficient vectors. They are calculated according to [1].
        :param epoch: the current epoch.
        :param position: the position of the current agent.
        :return: the coefficient vectors.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        n_dims = self._task.space_dimension
        a2 = -1. + epoch * (-1. / self._config.max_cycles)
        u = np.random.standard_normal(n_dims)
        # Coefficient vector for the update of the position of the agents
        cofi = np.zeros((4, n_dims))
        cofi[0, :] = np.random.random(n_dims)
        cofi[1, :] = (a2 + 1) + np.random.random()
        cofi[2, :] = a2 * np.random.standard_normal(n_dims)
        cofi[3, :] = u * np.random.standard_normal(n_dims) ** 2 * np.cos((np.random.random() * 2) * u)

        A = np.random.standard_normal(n_dims) * np.exp(2 - (epoch + 1) * (2. / self._config.max_cycles))
        D = (np.abs(position) + np.abs(self._best_agent.position)) * (2 * np.random.random() - 1)

        return cofi, A, D

    def __implement_candidate_positions__(
        self, cofi: np.ndarray, A: np.ndarray, D: np.ndarray, M: np.ndarray, position: np.ndarray
    ) -> list[np.ndarray]:
        """
        Implement the candidate positions. It is used to generate a new solution. It is a random combination of the
        best solution and the mean of the positions of the best solutions.
        :param cofi: the coefficient vector.
        :param A: the coefficient vector.
        :param D: the coefficient vector.
        :param M: the random solution.
        :param position: the position of the current agent.
        :return: the candidate positions.
        :rtype: list[np.ndarray]
        """
        best_position = np.array(self._best_agent.position)
        random_position = np.array(self._population[np.random.randint(0, self._config.population_size)].position)

        x1 = self._uniform_position()

        x2 = best_position - np.abs(
            (np.random.randint(1, 3) * M - np.random.randint(1, 3) * position) * A
        ) * cofi[np.random.randint(0, 4)]

        x3 = M + cofi[np.random.randint(0, 4)] + (
            np.random.randint(1, 3) * best_position - np.random.randint(1, 3) * random_position
        ) * cofi[np.random.randint(0, 4)]

        x4 = position - D + (
            np.random.randint(1, 3) * best_position - np.random.randint(1, 3) * M
        ) * cofi[np.random.randint(0, 4)]

        return [x1, x2, x3, x4]

    def optimization_step(self):
        pop_new = []
        for index in range(0, self._config.population_size):
            current_position = np.array(self._population[index].position)

            # get a new random solution
            M = self.__random_solution__()

            # calculate the vector of coefficients
            cofi, A, D = self.__coefficient_vectors__(self._cycles, current_position)

            # update the new population for the next generation with the calculated candidate positions
            pop_new += [
                self._init_agent(x) for x in self.__implement_candidate_positions__(cofi, A, D, M, current_position)
            ]

        self._extend_and_trim_population(pop_new)
