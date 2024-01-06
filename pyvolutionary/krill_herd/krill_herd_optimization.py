from typing import Final
import numpy as np

from ..helpers import (
    distance,
    special_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Krill, KrillHerdOptimizationConfig


class KrillHerdOptimization(OptimizationAbstract):
    """
    Implementation of the Krill Herd Optimization algorithm.

    Args:
        config (KrillHerdOptimizationConfig): an instance of KrillHerdOptimizationConfig class.
            {parse_obj_doc(KrillHerdOptimizationConfig)}

    Bibliography
    ----------
    [1] Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications
        in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704,
        https://doi.org/10.1016/j.cnsns.2012.05.010.
    """
    EPS: Final[float] = np.finfo(float).eps

    def __init__(self, config: KrillHerdOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__w_neighbour: np.ndarray | None = None
        self.__w_foraging: np.ndarray | None = None

    def before_initialization(self):
        self.__w_neighbour = np.full(self._task.space_dimension, self._config.w_neighbour)
        self.__w_foraging = np.full(self._task.space_dimension, self._config.w_foraging)

    def _init_agent(
        self,
        position: list[float] | np.ndarray | None = None,
        induced_speed: list[float] | None = None,
        foraging_speed: list[float] | None = None,
    ) -> Krill:
        agent = super()._init_agent(position)

        zeros = np.zeros(self._task.space_dimension).tolist()

        return Krill(
            **agent.model_dump(),
            induced_speed=induced_speed if induced_speed is not None else zeros,
            foraging_speed=foraging_speed if foraging_speed is not None else zeros,
        )

    def __get_food_location__(self) -> list[float]:
        """
        Get food location for krill heard.
        :return: food location.
        :rtype: list[float]
        """
        costs = np.array([krill.cost for krill in self._population])
        positions = np.array([krill.position for krill in self._population])

        return self._correct_position(np.array(
            [np.sum(positions[:, i] / costs) for i in range(0, self._task.space_dimension)]
        ) / np.sum(1 / costs))

    def __get_x__(self, x: Krill, y: Krill) -> np.ndarray:
        """
        Get x values.
        :param x: First krill/individual.
        :param y: Second krill/individual.
        :return: X values.
        :rtype: np.ndarray
        """
        return (
            (np.array(y.position) - np.array(x.position)) + self.EPS
        ) / (distance(y.position, x.position) + self.EPS)

    def __get_k__(self, x: Krill, y: Krill, best: Krill, worst: Krill) -> float:
        """
        Get k values.
        :param x: First krill/individual.
        :param y: Second krill/individual.
        :param best: Best krill/individual.
        :param worst: Worst krill/individual.
        :return: K values.
        :rtype: float
        """
        return (x.cost - y.cost + self.EPS) / (worst.cost - best.cost + self.EPS)

    def __induce_neighbours_motion__(self, idx: int, krill: Krill, best: Krill, worst: Krill) -> np.ndarray:
        """
        Induced neighbours motion operator.
        :param idx: Index of current krill being operated.
        :param krill: Krill being operated.
        :param best: Best krill in heard.
        :param worst: Worst krill in heard.
        :return: Induced speed.
        :rtype: np.ndarray
        """
        # calculate sense range for selected individual
        population_size = self._config.population_size
        sense_range = np.sum(
            [distance(krill.position, self._population[i].position) for i in range(0, population_size)]
        ) / (self._config.max_neighbours * population_size)

        # get neighbours
        neighbours = [agent for jdx, agent in enumerate(self._population) if jdx != idx and sense_range > distance(
            krill.position, agent.position
        )]
        if not neighbours:
            neighbours = [self._population[np.random.randint(self._config.population_size)]]

        alpha_l = np.sum(
            np.array([self.__get_k__(krill, neighbour, best, worst) for neighbour in neighbours]) * np.array(
                [self.__get_x__(krill, neighbour) for neighbour in neighbours]
            ).T
        )

        alpha_t = 2 * (1 + np.random.random() * (self._cycles + 1) / self._config.max_cycles)
        return self._config.n_max * (alpha_l + alpha_t) + self.__w_neighbour * krill.induced_speed

    def __induce_foraging_motion__(
            self, krill: Krill, pos_food: list[float], g_best: Krill, g_worst: Krill
    ) -> np.ndarray:
        """
        Induced foraging motion operator.
        :param krill: Krill being operated.
        :param pos_food: Food position.
        :param g_best: Best krill/individual.
        :param g_worst: Worst krill/individual.
        :return: Foraging speed.
        :rtype: np.ndarray
        """
        temp_krill = self._init_agent(position=pos_food)
        beta_f = 2 * (1 - (self._cycles + 1) / self._config.max_cycles) * self.__get_k__(
            krill, temp_krill, g_best, g_worst
        ) * self.__get_x__(krill, temp_krill) if g_best.cost < krill.cost else 0
        beta_b = self.__get_k__(krill, g_best, g_best, g_worst) * self.__get_x__(krill, g_best)
        return self._config.foraging_speed * (beta_f + beta_b) + self.__w_foraging * krill.foraging_speed

    def optimization_step(self):
        (g_best, ), (g_worst, ) = special_agents(self._population, n_best=1, n_worst=1)
        pos_food = self.__get_food_location__()

        new_pop = []
        for idx, krill in enumerate(self._population):
            pos = krill.position

            induced_speed = self.__induce_neighbours_motion__(idx, krill, g_best, g_worst)
            foraging_speed = self.__induce_foraging_motion__(krill, pos_food, g_best, g_worst)

            # induced physical diffusion operator
            diffusion = self._config.diffusion_speed * (
                1 - (self._cycles + 1) / self._config.max_cycles
            ) * np.random.uniform(-1, 1, self._task.space_dimension)

            # get a new position by a new delta factor
            new_pos = self._correct_position(krill.position + (
                self._config.c_t * np.sum(self._bandwidth()) * (induced_speed + foraging_speed + diffusion)
            ))

            # crossover
            crossover_rate = self._config.crossover_rate * self.__get_k__(krill, g_best, g_best, g_worst)
            new_pos = self._correct_position(np.where(
                [np.random.random() < crossover_rate for _ in range(0, self._task.space_dimension)], pos, new_pos
            ))

            # mutation
            mutation_rate = self._config.mutation_rate / (self.__get_k__(krill, g_best, g_best, g_worst) + 1e-31)
            new_pos = np.where(
                [np.random.random() < mutation_rate for _ in range(0, self._task.space_dimension)],
                new_pos,
                g_best.position + np.random.random(self._task.space_dimension)
            )

            new_pop.append(self._init_agent(new_pos, induced_speed.tolist(), foraging_speed.tolist()))

        self._greedy_select_population(new_pop)
