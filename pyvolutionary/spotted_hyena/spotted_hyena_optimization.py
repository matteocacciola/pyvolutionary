from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import SpottedHyenaOptimizationConfig, SpottedHyena


class SpottedHyenaOptimization(OptimizationAbstract):
    """
    Implementation of the Spotted Hyena Optimization algorithm.

    Args:
        config (SpottedHyenaOptimizationConfig): an instance of SpottedHyenaOptimizationConfig class.
            {parse_obj_doc(SpottedHyenaOptimizationConfig)}

    Bibliography
    ----------
    [1] Dhiman, G. and Kumar, V., 2017. Spotted hyena optimizer: a novel bio-inspired based metaheuristic
        technique for engineering applications. Advances in Engineering Software, 114, pp.48-70.
    """
    def __init__(self, config: SpottedHyenaOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__population_delta: list[SpottedHyena] | None = None
        self.__radius: np.ndarray | None = None
        self.__delta_max: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = SpottedHyenaOptimizationConfig(**parameters)

    def optimization_step(self):
        def get_n() -> int:
            done = False
            N = i = 0
            while not done and i < n_trials:
                pos_temp = np.array(g_best.position) + np.random.normal(0, 1, n_dims) * self._task.empty_solution()
                N += 1
                i += 1
                done = SpottedHyena(**self._init_agent(pos_temp).model_dump()).cost < g_best.cost
            return N + 1

        def circle_list_item(idx: int, B: np.ndarray, E: np.ndarray) -> np.ndarray:
            D_h = np.abs(np.dot(B, np.array(g_best.position)) - np.array(self._population[idx].position))
            p_k = np.array(g_best.position) - np.dot(E, D_h)
            return p_k

        def evolve(agent: SpottedHyena) -> SpottedHyena:
            pos = np.array(agent.position)
            B = 2 * np.random.uniform(0, 1, n_dims)
            E = hh * (2 * np.random.uniform(0, 1, n_dims) - 1)
            if np.random.random() < 0.5:
                D_h = np.abs(np.dot(B, np.array(g_best.position)) - pos)
                pos_new = np.array(g_best.position) - np.dot(E, D_h)
                new_agent = SpottedHyena(**self._init_agent(pos_new).model_dump())
                return self._greedy_select_agent(agent, new_agent)
            N = get_n()
            idx_list = np.random.choice(range(0, pop_size), N, replace=False).tolist()
            circle_list = [circle_list_item(idx_list[j], B, E) for j in range(0, N)]
            pos_new = np.mean(np.array(circle_list), axis=0)
            new_agent = SpottedHyena(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(agent, new_agent)

        n_trials = self._config.n_trials
        n_dims = self._task.space_dimension
        hh = self._config.h_factor - self._current_cycle * (self._config.h_factor / self._config.max_cycles)
        g_best = self._best_agent
        pop_size = self._config.population_size

        self._population = [evolve(agent) for agent in self._population]
