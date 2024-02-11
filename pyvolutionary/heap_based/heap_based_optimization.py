from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import HeapBasedOptimizationConfig


class HeapBasedOptimization(OptimizationAbstract):
    """
    Implementation of the Heap Based Optimization algorithm.

    Args:
        config (HeapBasedOptimizationConfig): an instance of HeapBasedOptimizationConfig class.
            {parse_obj_doc(HeapBasedOptimizationConfig)}

    Bibliography
    ----------
    [1] Askari, Q., Saeed, M., & Younas, I. (2020). Heap-based optimizer inspired by corporate rank hierarchy for global
        optimization. Expert Systems with Applications, 161, 113702.
    """
    def __init__(self, config: HeapBasedOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__it_per_cycle: float | None = None
        self.__qtr_cycle: float | None = None
        self.__heap: list[list[int | float]] | None = None
        self.__friend_limits: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = HeapBasedOptimizationConfig(**parameters)

    def before_initialization(self):
        cycles = np.floor(self._config.max_cycles / 25)
        self.__it_per_cycle = self._config.max_cycles / (cycles + self.EPS)
        self.__qtr_cycle = self.__it_per_cycle / 4

    def __heapifying__(self) -> list[list[int]]:
        pop_size = self._config.population_size
        degree = self._config.degree

        heap = []
        for c in range(pop_size):
            heap.append([self._population[c].cost, c])
            # Heapifying
            t = c
            while t > 0:
                parent_id = int(np.floor((t + 1) / degree) - 1)
                if self._population[parent_id].cost < self._population[t].cost:
                    break
                else:
                    heap[t], heap[parent_id] = heap[parent_id], heap[t]
                t = parent_id
        return heap

    def __colleagues_limits_generator__(self) -> np.ndarray:
        pop_size = self._config.population_size
        degree = self._config.degree

        friend_limits = np.zeros((pop_size, 2))
        for c in range(pop_size - 1, -1, -1):
            hi = int(np.ceil((np.log10(c * degree - c + 1) / np.log10(degree)))) - 1
            lower_lim = ((degree * degree ** (hi - 1) - 1) / (degree - 1) + 1)
            upper_lim = (degree * degree ** hi - 1) / (degree - 1)
            friend_limits[c, 0] = lower_lim if lower_lim <= pop_size else pop_size
            friend_limits[c, 1] = upper_lim if upper_lim <= pop_size else pop_size
        return friend_limits.astype(int)

    def after_initialization(self):
        self.__heap = self.__heapifying__()
        self.__friend_limits = self.__colleagues_limits_generator__()

    def optimization_step(self):
        epoch = self._current_cycle
        n_dims = self._task.space_dimension

        gama = (np.mod(epoch, self.__it_per_cycle) + 1) / self.__qtr_cycle
        gama = np.abs(2 - gama)
        p1 = 1. - epoch / self._config.max_cycles
        p2 = p1 + (1 - p1) / 2
        
        degree = self._config.degree

        for idx in range(self._config.population_size - 1, 0, -1):
            if idx == 0:  # root
                continue
            parent_id = int(np.floor((idx + 1) / degree) - 1)
            friend_idx = self.__friend_limits[idx, 0] if (
                self.__friend_limits[idx, 0] < self.__friend_limits[idx, 1] + 1
            ) else np.random.choice(
                list(set(range(self.__friend_limits[idx, 0], self.__friend_limits[idx, 1])) - {idx})
            )
            cur_agent_pos = np.array(self._population[self.__heap[idx][1]].position)  # solution to be updated
            # solutions to be updated with reference to
            par_agent_pos = np.array(self._population[self.__heap[parent_id][1]].position)
            fri_agent_pos = np.array(self._population[self.__heap[friend_idx][1]].position)
            # Position Updating
            rr = np.random.random(n_dims)
            rn = (2 * np.random.random(n_dims) - 1)
            first_option = par_agent_pos + rn * gama * np.abs(par_agent_pos - cur_agent_pos)
            second_option = fri_agent_pos + rn * gama * np.abs(fri_agent_pos - cur_agent_pos)
            third_option = cur_agent_pos + rn * gama * np.abs(fri_agent_pos - cur_agent_pos)
            new_pos = np.where(
                rr < p1,
                cur_agent_pos,
                np.where(
                    rr < p2,
                    first_option,
                    np.where(self.__heap[friend_idx][0] < self.__heap[idx][0], second_option, third_option)
                )
            )
            cur_agent = self._init_agent(new_pos)
            if cur_agent.cost < self.__heap[idx][0]:
                self._population[self.__heap[idx][1]] = cur_agent
                self.__heap[idx][0] = cur_agent.cost
            # heapifying
            t = idx
            while t > 1:
                parent_id = int((t + 1) / degree)
                if self.__heap[parent_id][0] < self.__heap[t][0]:
                    break
                else:
                    self.__heap[t], self.__heap[parent_id] = self.__heap[parent_id], self.__heap[t]
                t = parent_id
