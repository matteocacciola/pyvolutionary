from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import SineCosineAlgorithmOptimizationConfig, Candidate


class SineCosineAlgorithmOptimization(OptimizationAbstract):
    """
    Implementation of the Since Cosine Algorithm Optimization.

    Args:
        config (SineCosineAlgorithmOptimizationConfig): an instance of SineCosineAlgorithmOptimizationConfig class.
            {parse_obj_doc(SineCosineAlgorithmOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., 2016. SCA: a sine cosine algorithm for solving optimization problems. Knowledge-based systems,
        96, pp.120-133.
    """
    def __init__(self, config: SineCosineAlgorithmOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = SineCosineAlgorithmOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(idx: int, candidate: Candidate) -> Candidate:
            pos = np.array(candidate.position)
            # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
            r2 = 2 * np.pi * np.random.uniform(0, 1, n_dims)
            r3 = 2 * np.random.uniform(0, 1, n_dims)
            # Eq. 3.3, 3.1 and 3.2
            pos_new1 = pos + r1 * np.sin(r2) * np.abs(r3 * best_pos - pos)
            pos_new2 = pos + r1 * np.cos(r2) * np.abs(r3 * best_pos - pos)
            pos_new = np.where(np.random.random(n_dims) < 0.5, pos_new1, pos_new2)
            return self._greedy_select_agent(Candidate(**self._init_agent(pos_new).model_dump()), candidate)

        # Eq 3.4, r1 decreases linearly from "a" to 0
        r1 = 2.0 - (self._current_cycle + 1) * (2.0 / self._config.max_cycles)
        n_dims = self._task.space_dimension

        best_pos = np.array(self._best_agent.position)
        self._population = [evolve(idx, candidate) for idx, candidate in enumerate(self._population)]
