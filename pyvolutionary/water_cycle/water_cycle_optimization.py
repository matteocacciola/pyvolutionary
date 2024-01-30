from itertools import chain
from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    distance,
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Stream, WaterCycleOptimizationConfig


class WaterCycleOptimization(OptimizationAbstract):
    """
    Implementation of the Water Cycle Optimization algorithm.

    Args:
        config (WaterCycleOptimizationConfig): an instance of WaterCycleOptimizationConfig class.
            {parse_obj_doc(WaterCycleOptimizationConfig)}

    Bibliography
    ----------
    [1] Eskandar, H., Sadollah, A., Bahreininejad, A. and Hamdi, M., 2012. Water cycle algorithm–A novel metaheuristic
        optimization method for solving constrained engineering optimization problems. Computers & Structures, 110,
        pp.151-166.
    """
    def __init__(self, config: WaterCycleOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__ecc = 1e-6  # Evaporation condition
        self.__pop_best: list[Stream] = []  # Including sea and river (1st solution is sea)
        self.__streams: dict = {}

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = WaterCycleOptimizationConfig(**parameters)

    def after_initialization(self):
        n_stream = self._config.population_size - self._config.nsr
        self.__pop_best = self._population[:self._config.nsr]
        pop_stream = self._population[self._config.nsr:]  # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array([agent.cost for agent in self.__pop_best])
        num_child_in_river_list = np.round(np.abs(cost_river_list / np.sum(cost_river_list)) * n_stream).astype(int)
        if np.sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - np.sum(num_child_in_river_list)
        self.__streams = {}
        idx_already_selected = []
        for i in range(0, self._config.nsr - 1):
            idx_list = np.random.choice(
                list(set(range(0, n_stream)) - set(idx_already_selected)),
                num_child_in_river_list[i],
                replace=False
            ).tolist()
            idx_already_selected += idx_list

            self.__streams[i] = [pop_stream[idx] for idx in idx_list]
        idx_last = list(set(range(0, n_stream)) - set(idx_already_selected))
        self.__streams[self._config.nsr - 1] = [pop_stream[idx] for idx in idx_last]

    def optimization_step(self):
        def evolve_stream(idx: int, stream: Stream) -> Stream:
            pos = np.array(stream.position)
            pos_new = pos + np.random.uniform() * wc * (np.array(self.__pop_best[idx].position) - pos)
            return Stream(**self._init_agent(pos_new).model_dump())

        nsr = self._config.nsr
        wc = self._config.wc

        best_agent_pos = self._best_agent.position

        self.__streams = {
            idx: list(map(lambda s: evolve_stream(idx, s), streams)) for idx, streams in self.__streams.items()
        }
        self.__pop_best = [self._greedy_select_agent(
            best_agent(self.__streams[idx]), stream
        ) for idx, stream in enumerate(self.__pop_best)]

        # Evaporation
        evaporation_indexes = [idx for idx in range(1, nsr) if distance(
            best_agent_pos, self.__pop_best[idx].position
        ) < self.__ecc or np.random.random() < 0.1]
        for idx in evaporation_indexes:
            pop_current_best = sort_by_cost(self.__streams[idx] + [Stream(**self._init_agent().model_dump())])
            self.__pop_best[idx] = pop_current_best.pop(0)
            self.__streams[idx] = pop_current_best

        self._population = self.__pop_best.copy() + list(chain.from_iterable(self.__streams.values()))
        # Reduce the ecc
        self.__ecc = self.__ecc - self.__ecc / self._config.max_cycles
