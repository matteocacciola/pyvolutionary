from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import ArchimedeOptimizationConfig, Object


class ArchimedeOptimization(OptimizationAbstract):
    """
    Implementation of the Archimede Optimization algorithm.

    Args:
        config (ArchimedeOptimizationConfig): an instance of ArchimedeOptimizationConfig class.
            {parse_obj_doc(ArchimedeOptimizationConfig)}

    Bibliography
    ----------
    [1] Hashim, F.A., Hussain, K., Houssein, E.H., Mabrouk, M.S. and Al-Atabany, W., 2021. Archimedes optimization
        algorithm: a new metaheuristic algorithm for solving optimization problems. Applied Intelligence, 51(3),
        pp.1531-1551.
    """
    def __init__(self, config: ArchimedeOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = ArchimedeOptimizationConfig(**parameters)

    def _init_agent(
        self,
        position: list[Any] | np.ndarray | None = None,
        density: list[Any] | np.ndarray | None = None,
        volume: list[Any] | np.ndarray | None = None,
        acceleration: list[Any] | np.ndarray | None = None,
    ) -> Object:
        agent = super()._init_agent(position=position)

        density = self._task.empty_solution() if density is None else (
            density.tolist() if isinstance(density, np.ndarray) else density
        )
        volume = self._task.empty_solution() if volume is None else (
            volume.tolist() if isinstance(volume, np.ndarray) else volume
        )
        acceleration = self._task.empty_solution() if acceleration is None else (
            acceleration.tolist() if isinstance(acceleration, np.ndarray) else acceleration
        )
        return Object(**agent.model_dump(), density=density, volume=volume, acceleration=acceleration)

    def optimization_step(self):
        def calculate_new_properties(idx: int, obj: Object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            obj_density = np.array(obj.density)
            g_best_density = np.array(g_best.density)
            new_density = obj_density + np.random.uniform() * (g_best_density - obj_density)
            obj_volume = np.array(obj.volume)
            g_best_volume = np.array(g_best.volume)
            new_volume = obj_volume + np.random.uniform() * (g_best_volume - obj_volume)
            g_best_acceleration = np.array(g_best.acceleration)
            # Exploration phase
            if tf <= 0.5:
                # Update acceleration using Eq. 10 and normalize acceleration using Eq. 12
                id_rand = np.random.choice(list(set(range(0, pop_size)) - {idx}))
                new_acceleration = (np.array(self._population[id_rand].density) + np.array(
                    self._population[id_rand].volume
                )) * np.array(self._population[id_rand].acceleration) / (new_density * new_volume)
            else:
                new_acceleration = (g_best_density + g_best_volume * g_best_acceleration) / (
                    new_density * new_volume + self.EPS
                )
            return new_density, new_volume, new_acceleration

        def evolve(idx: int, obj: Object) -> Object:
            acc = config_max_acc * (new_accelerations[idx] - min_acc) / (max_acc - min_acc) + config_min_acc
            den = new_densities[idx]
            vol = new_volumes[idx]
            pos = np.array(obj.position)
            best_pos = np.array(g_best.position)
            if tf <= 0.5:  # update position using Eq. 13
                id_rand = np.random.choice(list(set(range(0, pop_size)) - {idx}))
                pos_new = pos + c1 * np.random.uniform() * acc * ddf * (
                    np.array(self._population[id_rand].position) - pos
                )
            else:
                f = 1 if (2 * np.random.random() - c4) <= 0.5 else -1
                t = c3 * tf
                pos_new = best_pos + f * c2 * np.random.random() * acc * ddf * (t * best_pos - pos)
            return self._greedy_select_agent(obj, self._init_agent(pos_new, den, vol, acc))

        pop_size = self._config.population_size
        tf = np.exp(self._current_cycle / self._config.max_cycles)
        # Density decreasing factor Eq. 9
        ddf = np.exp(1. - self._current_cycle / self._config.max_cycles) - self._current_cycle / self._config.max_cycles

        g_best = self._best_agent
        config_min_acc, config_max_acc = self._config.acc
        c1, c2, c3, c4 = self._config.c1, self._config.c2, self._config.c3, self._config.c4

        new_densities, new_volumes, new_accelerations = zip(*[
            calculate_new_properties(idx, obj) for idx, obj in enumerate(self._population)
        ])
        min_acc = np.min(new_accelerations)
        max_acc = np.max(new_accelerations)
        self._population = [evolve(idx, obj) for idx, obj in enumerate(self._population)]
