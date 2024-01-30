from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import WindDrivenOptimizationConfig, AirParcel


class WindDrivenOptimization(OptimizationAbstract):
    """
    Implementation of the Tuna Swarm Optimization algorithm.

    Args:
        config (WindDrivenOptimizationConfig): an instance of WindDrivenOptimizationConfig class.
            {parse_obj_doc(WindDrivenOptimizationConfig)}

    Bibliography
    ----------
    [1] Bayraktar, Z., Komurcu, M., Bossard, J.A. and Werner, D.H., 2013. The wind driven optimization technique and its
        application in electromagnetics. IEEE transactions on antennas and propagation, 61(5), pp.2745-2757.
    """
    def __init__(self, config: WindDrivenOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__dyn_list_velocity: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = WindDrivenOptimizationConfig(**parameters)

    def after_initialization(self):
        self.__dyn_list_velocity = self._config.max_v * np.array([
            self._task.initial_solution() for _ in range(self._config.population_size)
        ])

    def optimization_step(self):
        def evolve(idx: int, wind: AirParcel) -> tuple[list[float], AirParcel]:
            pos = np.array(wind.position)
            rand_dim = np.random.randint(0, n_dims)
            temp = self.__dyn_list_velocity[idx][rand_dim] * np.ones(n_dims)
            vel = (1 - alp) * self.__dyn_list_velocity[idx] - g_c * pos + (1 - 1.0 / (idx + 1)) * RT * (
                best_pos - pos
            ) + c_e * temp / (idx + 1)
            vel = np.clip(vel, -max_v, max_v).tolist()
            # Update air parcel positions, check the bound and calculate pressure (fitness)
            self.__dyn_list_velocity[idx] = vel
            return vel, AirParcel(**self._init_agent(pos + vel).model_dump())

        RT = self._config.RT
        g_c = self._config.g_c
        max_v = self._config.max_v
        c_e = self._config.c_e
        alp = self._config.alp
        n_dims = self._task.space_dimension
        
        best_pos = np.array(self._best_agent.position)
        self.__dyn_list_velocity, new_population = map(
            lambda x: list(x), zip(*[evolve(idx, wind) for idx, wind in enumerate(self._population)])
        )
        self.__dyn_list_velocity = np.array(self.__dyn_list_velocity)
        self._greedy_select_population(new_population)
