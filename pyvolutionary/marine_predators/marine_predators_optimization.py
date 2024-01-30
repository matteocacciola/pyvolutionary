from typing import Any
import numpy as np

from ..helpers import (
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import MarinePredatorsOptimizationConfig, MarinePredator


class MarinePredatorsOptimization(OptimizationAbstract):
    """
    Implementation of the Marine Predators Optimization algorithm.

    Args:
        config (MarinePredatorsOptimizationConfig): an instance of MarinePredatorsOptimizationConfig class.
            {parse_obj_doc(MarinePredatorsOptimizationConfig)}

    Bibliography
    ----------
    [1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020). Marine Predators Algorithm: A
        nature-inspired metaheuristic. Expert systems with applications, 152, 113377.
    """
    def __init__(self, config: MarinePredatorsOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__FADS = 0.2
        self.__P = 0.5

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = MarinePredatorsOptimizationConfig(**parameters)

    def optimization_step(self):
        def get_new_position(idx: int, predator: MarinePredator) -> np.ndarray:
            pos = np.array(predator.position)
            R = np.random.random(n_dims)
            if epoch < epochs / 3:  # Phase 1 (Eq.12)
                step_size = RB[idx] * (best_pos - RB[idx] * pos)
                return pos + P * R * step_size
            if epochs / 3 < epoch < 2 * epochs / 3:  # Phase 2 (Eqs. 13 & 14)
                if idx > pop_size / 2:
                    step_size = RB[idx] * (RB[idx] * best_pos - pos)
                    return best_pos + P * CF * step_size
                step_size = RL[idx] * (best_pos - RL[idx] * pos)
                return pos + P * R * step_size
            # Phase 3 (Eq. 15)
            step_size = RL[idx] * (RL[idx] * best_pos - pos)
            return best_pos + P * CF * step_size

        def evolve(idx: int, predator: MarinePredator) -> MarinePredator:
            pos = np.array(predator.position)
            pos_new = self._task.correct_solution(get_new_position(idx, predator))
            if np.random.random() < FADS:
                u = np.where(np.random.random(n_dims) < FADS, 1, 0)
                pos_new += CF * (lb + np.random.random(n_dims) * bandwidth) * u
                return self._greedy_select_agent(predator, MarinePredator(**self._init_agent(pos_new).model_dump()))
            r = np.random.random()
            step_size = (FADS * (1 - r) + r) * (
                np.array(self._population[per1[idx]].position) - np.array(self._population[per2[idx]].position)
            )
            pos_new += step_size
            return self._greedy_select_agent(predator, MarinePredator(**self._init_agent(pos_new).model_dump()))

        epoch = self._current_cycle
        epochs = self._config.max_cycles
        n_dims = self._task.space_dimension
        pop_size = self._config.population_size

        lb, _ = self._task.get_bounds()
        bandwidth = self._task.bandwidth()

        P, FADS = self.__P, self.__FADS

        CF = (1 - epoch / epochs) ** (2 * epoch / epochs)
        RL = get_levy_flight_step(beta=1.5, multiplier=0.05, size=(pop_size, n_dims), case=-1)
        RB = np.random.standard_normal(size=(pop_size, n_dims))
        per1 = np.random.permutation(pop_size)
        per2 = np.random.permutation(pop_size)

        best_pos = np.array(self._best_agent.position)

        self._population = [evolve(idx, predator) for idx, predator in enumerate(self._population)]
