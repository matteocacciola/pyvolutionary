from itertools import chain
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import MountainGazelleOptimizationConfig, MountainGazelle


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

    def optimization_step(self):
        def random_solution() -> np.ndarray:
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

        def coefficient_vectors(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = np.random.standard_normal(n_dims)
            # Coefficient vector for the update of the position of the agents
            cofi = np.zeros((4, n_dims))
            cofi[0, :] = np.random.random(n_dims)
            cofi[1, :] = (a2 + 1) + np.random.random()
            cofi[2, :] = a2 * np.random.standard_normal(n_dims)
            cofi[3, :] = u * np.random.standard_normal(n_dims) ** 2 * np.cos((np.random.random() * 2) * u)
            D = (np.abs(position) + np.abs(best_position)) * (2 * np.random.random() - 1)
            return cofi, D

        def candidate_positions(cofi: np.ndarray, D: np.ndarray, position: np.ndarray) -> list[np.ndarray]:
            random_position = np.array(
                self._population[np.random.randint(0, self._config.population_size)].position
            )
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

        def new_population(gazelle: MountainGazelle) -> list[MountainGazelle]:
            current_position = np.array(gazelle.position)
            # get a new random solution
            # calculate the vector of coefficients
            cofi, D = coefficient_vectors(current_position)
            candidates = candidate_positions(cofi, D, current_position)
            # update the new population for the next generation with the calculated candidate positions
            return [MountainGazelle(**self._init_agent(x).model_dump()) for x in candidates]

        epoch = self._current_cycle
        n_dims = self._task.space_dimension
        pop_size = self._config.population_size

        a2 = -1. + epoch * (-1. / self._config.max_cycles)

        best_position = np.array(self._best_agent.position)
        A = np.random.standard_normal(n_dims) * np.exp(2 - (epoch + 1) * (2. / self._config.max_cycles))
        M = random_solution()

        pop_new = list(chain.from_iterable([new_population(gazelle) for gazelle in self._population]))
        self._extend_and_trim_population(pop_new)
