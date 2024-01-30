from itertools import chain
from typing import Any
import numpy as np

from ..helpers import (
    find_centers,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Person, BrainStormOptimizationConfig


class BrainStormOptimization(OptimizationAbstract):
    """
    Implementation of the Brain Storm Optimization algorithm.

    Args:
        config (BrainStormOptimizationConfig): an instance of BrainStormOptimizationConfig class.
            {parse_obj_doc(BrainStormOptimizationConfig)}

    Bibliography
    ----------
    [1] Shi, Y., 2011, June. Brain storm optimization algorithm. In International conference in swarm intelligence
        (pp. 303-309). Springer, Berlin, Heidelberg.
    """
    def __init__(self, config: BrainStormOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__m_solution: int | None = None
        self.__clusters: list[list[Person]] = []
        self.__centers: list[Person] = []

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = BrainStormOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__m_solution = int(self._config.population_size / self._config.m_clusters)

    def after_initialization(self):
        self.__clusters = self._generate_group_population(self._config.m_clusters, self.__m_solution, False)
        self.__centers: list[Person] = find_centers(self.__clusters)

    def optimization_step(self):
        def evolve(idx: int) -> tuple[Person, int, int]:
            cluster_id = int(idx / m_solution)
            location_id = int(idx % m_solution)
            if np.random.uniform() < p2:  # p_6b
                rand_idx = np.random.randint(0, m_solution)
                pos_new = np.array(self.__clusters[cluster_id][rand_idx].position) + np.random.normal(0, 1, n_dims)
                if np.random.uniform() < p3:  # p_6i
                    cluster_id = np.random.randint(0, m_clusters)
                    pos_new = np.array(self.__centers[cluster_id].position) + epsilon * np.random.normal(0, 1, n_dims)
            else:
                id1, id2 = np.random.choice(range(0, m_clusters), 2, replace=False)
                pos_new = 0.5 * (
                    np.array(self.__centers[id1].position) + np.array(self.__centers[id2].position)
                ) + epsilon * np.random.normal(0, 1, n_dims)
                if np.random.uniform() >= p4:
                    pos_rand_id1 = np.array(self.__clusters[id1][np.random.randint(0, m_solution)].position)
                    pos_rand_id2 = np.array(self.__clusters[id2][np.random.randint(0, m_solution)].position)
                    pos_new = 0.5 * (pos_rand_id1 + pos_rand_id2) + epsilon * np.random.normal(0, 1, n_dims)
            agent = Person(**self._init_agent(pos_new).model_dump())
            return agent, cluster_id, location_id

        epoch = self._current_cycle
        epochs = self._config.max_cycles

        x = (0.5 * epochs - epoch) / self._config.slope
        epsilon = np.random.uniform() * (1 / (1 + np.exp(-x)))

        p1, p2, p3, p4 = self._config.p1, self._config.p2, self._config.p3, self._config.p4
        m_solution = self.__m_solution

        n_dims = self._task.space_dimension
        m_clusters = self._config.m_clusters

        epsilon = 1. - 1. * epoch / epochs
        if np.random.uniform() < p1:
            self.__centers[np.random.randint(0, m_clusters)] = Person(**self._init_agent().model_dump())

        new_agents, cluster_ids, location_ids = zip(*[evolve(idx) for idx in range(0, self._config.population_size)])
        for cl_id, loc_id, new_agent in zip(cluster_ids, location_ids, new_agents):
            self.__clusters[cl_id][loc_id] = self._greedy_select_agent(self.__clusters[cl_id][loc_id], new_agent)

        self.__centers = find_centers(self.__clusters)
        self._population = list(chain.from_iterable(self.__clusters))
