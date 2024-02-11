from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import SalpSwarmOptimizationConfig, Salp


class SalpSwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Salp Swarm Optimization algorithm.

    Args:
        config (SalpSwarmOptimizationConfig): an instance of SalpSwarmOptimizationConfig class.
            {parse_obj_doc(SalpSwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z., Saremi, S., Faris, H. and Mirjalili, S.M., 2017. Salp Swarm
        Algorithm: A bio-inspired optimizer for engineering design problems. Advances in Engineering Software, 114,
        pp.163-191.
    """
    def __init__(self, config: SalpSwarmOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = SalpSwarmOptimizationConfig(**parameters)

    def optimization_step(self):
        def new_position(idx: int, salp: Salp) -> np.ndarray:
            if idx < pop_size / 2:
                c2_list = np.random.random(n_dims)
                c3_list = np.random.random(n_dims)
                return np.where(
                    c3_list < 0.5,
                    best_pos + c1 * (bandwidth * c2_list + lb),
                    best_pos - c1 * (bandwidth * c2_list + lb),
                )
            # Eq. (3.4) in the paper
            return (np.array(salp.position) + np.array(self._population[idx - 1].position)) / 2

        def evolve(idx: int, salp: Salp) -> Salp:
            pos_new = new_position(idx, salp)
            agent = Salp(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(agent, salp)

        n_dims = self._task.space_dimension
        cycle_ratio = self._current_cycle / self._config.max_cycles
        c1 = 2 * np.exp(-((4 * cycle_ratio) ** 2))
        pop_size = self._config.population_size

        best_pos = np.array(self._best_agent.position)
        bandwidth = self._task.bandwidth()
        lb, _ = self._task.get_bounds()

        self._population = [evolve(idx, salp) for idx, salp in enumerate(self._population)]
