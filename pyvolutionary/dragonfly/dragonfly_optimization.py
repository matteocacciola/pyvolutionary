from typing import Any
import numpy as np

from ..helpers import (
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import DragonflyOptimizationConfig, Dragonfly


class DragonflyOptimization(OptimizationAbstract):
    """
    Implementation of the Dragonfly Optimization algorithm.

    Args:
        config (DragonflyOptimizationConfig): an instance of DragonflyOptimizationConfig class.
            {parse_obj_doc(DragonflyOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., 2016. Dragonfly algorithm: a new meta-heuristic optimization technique for solving
        single-objective, discrete, and multi-objective problems. Neural computing and applications, 27(4),
        pp.1053-1073.
    """
    def __init__(self, config: DragonflyOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__population_delta: list[Dragonfly] | None = None
        self.__radius: np.ndarray | None = None
        self.__delta_max: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = DragonflyOptimizationConfig(**parameters)

    def after_initialization(self):
        self.__population_delta = self._generate_agents(self._config.population_size)
        # Initial radius of dragonflies' neighborhoods, and maximum delta value
        self.__radius = self.__delta_max = self._task.bandwidth() / 10

    def optimization_step(self):
        def evolve(dragonfly: Dragonfly, dragonfly_delta: Dragonfly) -> tuple[Dragonfly, Dragonfly]:
            pos = np.array(dragonfly.position)
            pos_delta = np.array(dragonfly_delta.position)
            # find the neighbouring solutions: those within the radius in each dimension, and not overlapping
            dist = np.abs(pos - positions)
            is_neighbour = np.all(dist <= r, axis=1) & np.all(dist != 0, axis=1)
            pos_neighbours = positions[is_neighbour]
            pos_neighbours_delta = positions_delta[is_neighbour]
            neighbours_num = len(pos_neighbours)
            # separation: Eq 3.1, Alignment: Eq 3.2, Cohesion: Eq 3.3
            S = np.zeros(n_dims)
            A = pos_delta.copy()
            C_temp = pos.copy()
            if neighbours_num > 1:
                S = np.sum(np.array(pos_neighbours), axis=0) - neighbours_num * pos
                A = np.sum(np.array(pos_neighbours_delta), axis=0) / neighbours_num
                C_temp = np.sum(pos_neighbours, axis=0) / neighbours_num
            C = C_temp - pos
            # attraction to food: Eq 3.4
            dist_to_food = np.abs(pos - g_best_position)
            F = g_best_position - pos if np.all(dist_to_food <= r) else np.zeros(n_dims)
            # distraction from enemy: Eq 3.5
            dist_to_enemy = np.abs(pos - g_worst_position)
            enemy = g_worst_position + pos if np.all(dist_to_enemy <= r) else np.zeros(n_dims)
            pos_new = pos.copy().astype(float)
            if np.any(dist_to_food > r):
                if neighbours_num > 1:
                    temp = temp_new = w * pos_delta + np.random.uniform(0, 1, n_dims) * A + (
                        np.random.uniform(0, 1, n_dims) * C + np.random.uniform(0, 1, n_dims) * S
                    )
                else:
                    temp = get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1) * pos  # Eq. 3.8
                    temp_new = np.zeros(n_dims)
            else:
                temp = temp_new = (a * A + c * C + s * S + f * F + e * enemy) + w * pos_delta  # Eq. 3.6
            pos_new += np.clip(temp, -1 * self.__delta_max, self.__delta_max)
            pos_delta_new = np.clip(temp_new, -1 * self.__delta_max, self.__delta_max)
            # amend solution
            agent_new = self._greedy_select_agent(dragonfly, Dragonfly(**self._init_agent(pos_new).model_dump()))
            agent_delta_new = self._greedy_select_agent(
                dragonfly_delta, Dragonfly(**self._init_agent(pos_delta_new).model_dump())
            )
            return agent_new, agent_delta_new

        bandwidth = self._task.bandwidth()
        cycle = self._current_cycle
        max_cycles = self._config.max_cycles
        n_dims = self._task.space_dimension
        
        r = bandwidth / 4 + (bandwidth * (2 * cycle / max_cycles))
        w = 0.9 - cycle * ((0.9 - 0.4) / max_cycles)
        my_c = 0.1 - cycle * ((0.1 - 0) / (max_cycles / 2))
        my_c = 0 if my_c < 0 else my_c

        s, a, c = 2 * np.random.random(3) * my_c  # separation, alignment and cohesion weight
        f = 2 * np.random.random()  # Food attraction weight
        e = my_c  # Enemy distraction weight

        g_best_position = np.array(self._best_agent.position)
        g_worst_position = np.array(self._worst_agent.position)

        positions = np.array([dragonfly.position for dragonfly in self._population])
        positions_delta = np.array([dragonfly.position for dragonfly in self.__population_delta])
        
        self._population, self.__population_delta = map(
            lambda x: list(x),
            zip(*[evolve(dragonfly, self.__population_delta[idx]) for idx, dragonfly in enumerate(self._population)]),
        )
