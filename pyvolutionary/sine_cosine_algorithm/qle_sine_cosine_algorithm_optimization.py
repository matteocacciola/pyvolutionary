from typing import Any
import numpy as np

from .classes import QTable
from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import QleSineCosineAlgorithmOptimizationConfig, QleCandidate


class QleSineCosineAlgorithmOptimization(OptimizationAbstract):
    """
    Implementation of the QLE Sine Cosine Algorithm Optimization.

    Args:
        config (QleSineCosineAlgorithmOptimizationConfig): an instance of QleSineCosineAlgorithmOptimizationConfig
            class.
            {parse_obj_doc(QleSineCosineAlgorithmOptimizationConfig)}

    Bibliography
    ----------
    [1] Hamad, Q. S., Samma, H., Suandi, S. A., & Mohamad-Saleh, J. (2022). Q-learning embedded sine cosine algorithm
        (QLESCA). Expert Systems with Applications, 193, 116417.
    """
    def __init__(self, config: QleSineCosineAlgorithmOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = QleSineCosineAlgorithmOptimizationConfig(**parameters)

    def _init_agent(self, position: list[Any] | np.ndarray | None = None) -> QleCandidate:
        agent = super()._init_agent(position)
        return QleCandidate(**agent.model_dump(), qtable=QTable(n_states=9, n_actions=9))

    def optimization_step(self):
        def evolve(candidate: QleCandidate) -> QleCandidate:
            pos = np.array(candidate.position)
            # Step 4: Action execution
            state = candidate.qtable.get_state(density=density, distance=distances)
            action = candidate.qtable.get_action(state=state)
            r1_bound, r3_bound = candidate.qtable.get_action_params(action)
            r1 = np.random.uniform(r1_bound[0], r1_bound[1])
            r3 = np.random.uniform(r3_bound[0], r3_bound[1])
            r2 = 2 * np.pi * np.random.uniform()
            r4 = np.random.uniform()
            pos_new = pos + r1 * (np.sin(r2) if r4 < 0.5 else np.cos(r2)) * (r3 * best_pos - pos)
            new_candidate = QleCandidate(**self._init_agent(pos_new).model_dump())
            new_candidate.qtable.update(state, action, reward=1, alpha=alpha, gama=gama)
            candidate.qtable.update(state, action, reward=-1, alpha=alpha, gama=gama)
            return self._greedy_select_agent(new_candidate, candidate)

        alpha, gama = self._config.alpha, self._config.gama

        best = self._best_agent
        best_pos = np.array(best.position)

        # Step 3: state computation
        # calculating the density of candidates
        positions = np.array([agent.position for agent in self._population])
        dists = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions) ** 2, axis=-1))
        density = 1 / (self._config.population_size * np.max(dists)) * np.sum(
            np.sqrt(np.sum((positions - np.mean(positions, axis=0)) ** 2, axis=1))
        )

        # calculate the distance
        distances = np.sum(
            np.sqrt(np.sum((best_pos - positions) ** 2, axis=1))
        ) / np.sum([np.sqrt(np.sum(self._task.bandwidth() ** 2)) for _ in range(0, self._config.population_size)])

        self._population = [evolve(zebra) for zebra in self._population]
