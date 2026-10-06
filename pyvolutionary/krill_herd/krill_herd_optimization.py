from typing import Any
import numpy as np

from ..helpers import (
    distance,
    per_row,
    squared_norms,
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
    def __init__(self, config: KrillHerdOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__w_neighbour: np.ndarray | None = None
        self.__w_foraging: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = KrillHerdOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__w_neighbour = np.full(self._task.space_dimension, self._config.w_neighbour)
        self.__w_foraging = np.full(self._task.space_dimension, self._config.w_foraging)

    def _init_agent(
        self,
        position: list[Any] | np.ndarray | None = None,
        induced_speed: list[float] | None = None,
        foraging_speed: list[float] | None = None,
    ) -> Krill:
        agent = super()._init_agent(position)

        zeros = np.zeros(self._task.space_dimension).tolist()

        return Krill(
            **agent.__dict__,
            induced_speed=induced_speed if induced_speed is not None else zeros,
            foraging_speed=foraging_speed if foraging_speed is not None else zeros,
        )

    def optimization_step(self):
        def get_x(x: Krill, y: Krill) -> np.ndarray:
            return (
                (np.array(y.position) - np.array(x.position)
            ) + self.EPS) / (distance(y.position, x.position) + self.EPS)

        def get_k(x: Krill, y: Krill) -> float:
            return (x.cost - y.cost + self.EPS) / (g_worst.cost - g_best.cost + self.EPS)

        def induce_neighbours_motion(idx: int, krill: Krill) -> np.ndarray:
            # calculate sense range for selected individual
            krill_distances = distances_matrix[idx]
            sense_range = np.sum(krill_distances) / (max_neighbours * population_size)
            # get neighbours
            neighbours_idx = [jdx for jdx in range(0, population_size) if jdx != idx and sense_range > krill_distances[jdx]]
            if not neighbours_idx:
                neighbours_idx = [np.random.randint(population_size)]
            # get_k and get_x of the krill with respect to each neighbour, vectorized over the neighbours
            neighbours_k = (krill.cost - costs[neighbours_idx] + self.EPS) / (g_worst.cost - g_best.cost + self.EPS)
            neighbours_x = (
                (positions[neighbours_idx] - positions[idx]) + self.EPS
            ) / per_row(distances_matrix[idx, neighbours_idx] + self.EPS, positions)
            alpha_l = np.sum(neighbours_k * neighbours_x.T)
            alpha_t = 2 * (1 + np.random.random() * (current_cycle + 1) / max_cycles)
            return n_max * (alpha_l + alpha_t) + w_neighbour * krill.induced_speed

        def induce_foraging_motion(krill: Krill) -> np.ndarray:
            beta_f = 2 * (1 - (current_cycle + 1) / max_cycles) * get_k(krill, temp_krill) * get_x(
                krill, temp_krill
            ) if g_best.cost < krill.cost else 0
            beta_b = get_k(krill, g_best) * get_x(krill, g_best)
            return config_foraging_speed * (beta_f + beta_b) + w_foraging * krill.foraging_speed

        def new_population(idx: int, krill: Krill) -> Krill:
            pos = krill.position
            # induced physical diffusion operator
            diffusion = config_diffusion_speed * (1 - (current_cycle + 1) / max_cycles) * np.random.uniform(-1, 1, dims)
            # get a new position by a new delta factor
            new_pos = self._task.correct_solution(
                krill.position + (c_t * sum_bandwidth * (induced_speed[idx] + foraging_speed[idx] + diffusion))
            )
            # crossover
            new_pos = self._task.correct_solution(np.where(
                np.random.random(dims) < config_crossover_rate * get_k(krill, g_best),
                pos,
                new_pos
            ))
            # mutation
            mutation_rate = config_mutation_rate / (get_k(krill, g_best) + 1e-31)
            new_pos = np.where(
                np.random.random(self._task.space_dimension) < mutation_rate,
                new_pos,
                g_best.position + np.random.random(self._task.space_dimension)
            )
            return self._init_agent(new_pos, induced_speed[idx].tolist(), foraging_speed[idx].tolist())

        dims = self._task.space_dimension
        population_size = self._config.population_size
        current_cycle, max_cycles = self._current_cycle, self._config.max_cycles

        (g_best, ), (g_worst, ) = special_agents(self._population, n_best=1, n_worst=1)

        config_diffusion_speed, config_foraging_speed = self._config.diffusion_speed, self._config.foraging_speed
        config_crossover_rate, config_mutation_rate = self._config.crossover_rate, self._config.mutation_rate
        max_neighbours = self._config.max_neighbours
        n_max = self._config.n_max
        c_t = self._config.c_t

        w_neighbour, w_foraging = self.__w_neighbour, self.__w_foraging

        # get food location
        costs = np.array([krill.cost for krill in self._population])
        positions = np.array([krill.position for krill in self._population])
        # distances between each pair of krills (Euclidean, as computed by the distance helper)
        distances_matrix = np.array([
            np.sqrt(squared_norms(positions - position)) for position in positions
        ])
        pos_food = np.sum(
            np.array([np.array(krill.position) / krill.cost for krill in self._population]),
            axis=0
        ) / np.sum(1 / costs)

        induced_speed = [induce_neighbours_motion(idx, krill) for idx, krill in enumerate(self._population)]
        # the food is evaluated once: the same position would be evaluated again for each krill
        temp_krill = self._init_agent(position=pos_food)
        foraging_speed = [induce_foraging_motion(krill) for krill in self._population]

        sum_bandwidth = np.sum(self._task.bandwidth())

        self._greedy_select_population([new_population(idx, krill) for idx, krill in enumerate(self._population)])
