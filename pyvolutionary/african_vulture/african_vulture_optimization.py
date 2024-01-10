import numpy as np

from ..helpers import (
    best_agents,
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import AfricanVulture, AfricanVultureOptimizationConfig


class AfricanVultureOptimization(OptimizationAbstract):
    """
    Implementation of the African Vulture Optimization algorithm.

    Args:
        config (AfricanVultureOptimizationConfig): an instance of AfricanVultureOptimizationConfig class.
            {parse_obj_doc(AfricanVultureOptimizationConfig)}

    Bibliography
    ----------
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new
        nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering,
        158, 107408.
    """
    def __init__(self, config: AfricanVultureOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def exploration_position_phase1(rand_pos: np.ndarray, position: np.ndarray) -> np.ndarray:
            best_x1 = np.array(best_list[0].position)
            best_x2 = np.array(best_list[1].position)
            if np.random.random() < p2:
                A = best_x1 - ((best_x1 * position) / (best_x1 - position ** 2)) * F
                B = best_x2 - ((best_x2 * position) / (best_x2 - position ** 2)) * F
                return (A + B) / 2
            return rand_pos - np.abs(rand_pos - position) * F * get_levy_flight_step(
                beta=1.5, multiplier=1., size=dim, case=-1
            )

        def exploration_position_phase2(rand_pos: np.ndarray, position: np.ndarray) -> np.ndarray:
            if np.random.random() < p3:
                return (
                    np.abs(2 * np.random.random() * rand_pos - position)
                ) * (F + np.random.random()) - (rand_pos - position)
            s1 = np.random.random() * position * np.cos(position)
            s2 = np.random.random() * position * np.sin(position)
            return rand_pos * (1 - ((s1 + s2) / (2 * np.pi)))

        def exploration_position(rand_pos: np.ndarray, position: np.ndarray) -> np.ndarray:
            if np.random.random() < p1:
                return rand_pos - (np.abs((2 * np.random.random()) * rand_pos - position)) * F
            return rand_pos - F + np.random.random() * self._uniform_position()

        def evolve(vulture: AfricanVulture) -> AfricanVulture:
            position = np.array(vulture.position)
            rand_pos = np.array(best_list[np.random.choice([0, 1], p=[alpha, 1 - alpha])].position)
            if np.abs(F) >= 1:  # Exploration
                pos_new = exploration_position(rand_pos, position)
            elif np.abs(F) < 0.5:  # Exploitation Phase 1
                pos_new = exploration_position_phase1(rand_pos, position)
            else:  # Exploitation Phase 2
                pos_new = exploration_position_phase2(rand_pos, position)
            return self._greedy_select_agent(
                vulture, AfricanVulture(**self._init_agent(pos_new).model_dump())
            )

        cycle_ratio = self._current_cycle / self._config.max_cycles
        p1, p2, p3 = self._config.p
        dim = self._task.space_dimension

        ppp = (2 * np.random.random() + 1) * (1 - cycle_ratio) + np.random.uniform(-2, 2) * (
            (np.sin((np.pi / 2) * cycle_ratio) ** self._config.gamma) + np.cos((np.pi / 2) * cycle_ratio) - 1
        )

        alpha = self._config.alpha
        F = ppp * (2 * np.random.random() - 1)
        best_list = best_agents(self._population, 2)

        self._population = [evolve(agent) for agent in self._population]
