from typing import Any
import numpy as np

from ..helpers import (
    roulette_wheel_indexes,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import AntLionOptimizationConfig, AntLion


class AntLionOptimization(OptimizationAbstract):
    """
    Implementation of the Ant Lion Optimization algorithm.

    Args:
        config (AntLionOptimizationConfig): an instance of AntLionOptimizationConfig class.
            {parse_obj_doc(AntLionOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., 2015. The ant lion optimizer. Advances in engineering software, 83, pp.80-98.
    """
    def __init__(self, config: AntLionOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = AntLionOptimizationConfig(**parameters)

    def optimization_step(self):
        def get_i_ratio() -> float:
            # I is the ratio in Equations (2.10) and (2.11)
            if cycle > max_cycles / 10:
                return 1 + 100 * (cycle / max_cycles)
            if cycle > max_cycles / 2:
                return 1 + 1000 * (cycle / max_cycles)
            if cycle > max_cycles * (3 / 4):
                return 1 + 10000 * (cycle / max_cycles)
            if cycle > max_cycles * 0.9:
                return 1 + 100000 * (cycle / max_cycles)
            if cycle > max_cycles * 0.95:
                return 1 + 1000000 * (cycle / max_cycles)
            return 1

        def random_walk_ant_lion(ant_lion_position: np.ndarray) -> np.ndarray:
            X = np.array([np.cumsum(2 * (np.random.random(pop_size) > 0.5) - 1) for _ in range(0, n_dims)])
            # Move the interval of [lb ub] around the ant lion [lb + ant_lion, ub + ant_lion]. Eq 2.8, 2.9
            lb = lower_bounds * (1 if np.random.random() < 0.5 else -1) + ant_lion_position
            ub = upper_bounds * (1 if np.random.random() < 0.5 else -1) + ant_lion_position
            a = np.min(X, axis=1)
            b = np.max(X, axis=1)
            return (X - np.reshape(a, (n_dims, 1))) * (
                np.reshape((ub - lb) / (b - a), (n_dims, 1))
            ) + np.reshape(lb, (n_dims, 1))

        def new_agent(idx: int) -> AntLion:
            roulette_index, = roulette_wheel_indexes(weights)
            # RA is the random walk around the selected ant lion by roulette wheel
            RA = random_walk_ant_lion(np.array(self._population[roulette_index].position))
            # RE is the random walk around the elite (the best ant lion so far)
            RE = random_walk_ant_lion(g_best_pos)
            pos_new = (RA[:, idx] + RE[:, idx]) / 2  # Equation(2.13) in the paper
            return AntLion(**self._init_agent(pos_new).model_dump())

        cycle = self._current_cycle
        max_cycles = self._config.max_cycles

        I = get_i_ratio()
        lower_bounds, upper_bounds = self._task.get_bounds()
        # Decrease boundaries to converge towards ant lion
        lower_bounds /= I  # Equation (2.10) in the paper
        upper_bounds /= I  # Equation (2.10) in the paper

        n_dims = self._task.space_dimension
        pop_size = self._config.population_size
        g_best_pos = np.array(self._best_agent.position)

        # Select ant lions based on their fitness (the better ant lion the higher chance of catching ant)
        weights = np.array([ant_lion.fitness for ant_lion in self._population])
        weights /= np.sum(weights)
        pop_new = [new_agent(idx) for idx in range(0, pop_size)]

        self._extend_and_trim_population(pop_new)
