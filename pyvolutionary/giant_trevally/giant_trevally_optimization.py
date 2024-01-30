from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import GiantTrevallyOptimizationConfig, GiantTrevally


class GiantTrevallyOptimization(OptimizationAbstract):
    """
    Implementation of the Giant Trevally Optimization algorithm.

    Args:
        config (GiantTrevallyOptimizationConfig): an instance of GiantTrevallyOptimizationConfig class.
            {parse_obj_doc(GiantTrevallyOptimizationConfig)}

    Bibliography
    ----------
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic Algorithm for
        Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121
    """
    def __init__(self, config: GiantTrevallyOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = GiantTrevallyOptimizationConfig(**parameters)

    def optimization_step(self):
        def extensive_search(trevally: GiantTrevally) -> GiantTrevally:
            pos_new = best_pos * np.random.random() + (
                (ub - lb) * np.random.random() + lb
            ) * get_levy_flight_step(beta=1.5, multiplier=0.01, size=n_dims, case=-1)
            return self._greedy_select_agent(trevally, GiantTrevally(**self._init_agent(pos_new).model_dump()))

        def choosing_area(trevally: GiantTrevally) -> GiantTrevally:
            pos = np.array(trevally.position)
            r3 = np.random.random()
            pos_new = best_pos * A * r3 + pos_m - pos * r3  # Eq. 7
            return self._greedy_select_agent(trevally, GiantTrevally(**self._init_agent(pos_new).model_dump()))

        def attacking(trevally: GiantTrevally) -> GiantTrevally:
            pos = np.array(trevally.position)
            # the distance between the prey and the attacker, and can be calculated using (12):
            dist = np.sum(np.abs(best_pos - pos))
            theta2 = 360 * np.random.random()
            theta1 = (1.33 / 1.00029) * np.sin(np.radians(theta2))  # calculate theta_1 using (10)
            # visual distortion indicates the apparent height of the bird, which is always seen
            # to be higher than its actual height due to the refraction of the light.
            VD = np.sin(np.radians(theta1)) * dist  # Eq. 11
            # the behavior of giant trevally when chasing and jumping out of the water is mathematically simulated
            pos_new = pos * np.sin(np.radians(theta2)) * trevally.cost + VD + H  # Eq. (13)
            return self._greedy_select_agent(trevally, GiantTrevally(**self._init_agent(pos_new).model_dump()))

        n_dims = self._task.space_dimension
        epoch = self._current_cycle
        epochs = self._config.max_cycles
        lb, ub = self._task.get_bounds()

        best_pos = np.array(self._best_agent.position)

        # foraging movement patterns of giant trevallies are simulated using Eq.(4)
        self._population = [extensive_search(trevally) for trevally in self._population]
        best_pos = np.array(best_agent(self._population).position)

        # in the pursuing step, giant trevallies pursue the best area in terms of the amount of food (seabirds)
        A = 0.4
        pos_m = np.mean(np.array([agent.position for agent in self._population]), axis=0)
        self._population = [choosing_area(trevally) for trevally in self._population]
        best_pos = np.array(best_agent(self._population).position)

        # attacking: in this step, giant trevallies attack seabirds in the best area
        H = np.random.random() * (2.0 - (epoch + 1) * 2.0 / epochs)  # Eq.(15)
        self._population = [attacking(trevally) for trevally in self._population]
