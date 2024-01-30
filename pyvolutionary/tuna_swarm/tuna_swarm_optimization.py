from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import TunaSwarmOptimizationConfig, Tuna


class TunaSwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Tuna Swarm Optimization algorithm.

    Args:
        config (TunaSwarmOptimizationConfig): an instance of TunaSwarmOptimizationConfig class.
            {parse_obj_doc(TunaSwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Xie, L., Han, T., Zhou, H., Zhang, Z. R., Han, B., & Tang, A. (2021). Tuna swarm optimization: a novel
        swarm-based metaheuristic algorithm for global optimization. Computational intelligence and Neuroscience.
    """
    def __init__(self, config: TunaSwarmOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = TunaSwarmOptimizationConfig(**parameters)

    def optimization_step(self):
        def get_new_local_pos(pos: list[float], prev_pos: list[float]) -> list[float] | None:
            pos = np.array(pos)
            prev_pos = np.array(prev_pos)
            if np.random.random() < zz:
                return None
            diff_pos = best_position - pos
            if 0.5 < np.random.random():
                r1 = np.random.random()
                beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((epochs - epoch + 1) / epochs)))) * np.cos(2 * np.pi * r1)
                if C > np.random.random():
                    return (a1 * (best_position + beta * np.abs(diff_pos)) + a2 * prev_pos).tolist()  # Eqs (8.3-8.4)
                rand_pos = np.array(self._task.initial_solution())
                return (a1 * (rand_pos + beta * np.abs(rand_pos - pos)) + a2 * prev_pos).tolist()  # Eqs (8.1-8.2)
            tf = np.random.choice([-1, 1])
            if 0.5 > np.random.random():
                return (
                    best_position + np.random.random(n_dims) * diff_pos + tf * tt ** 2 * diff_pos
                ).tolist()  # Eq 9.1
            return (tf * tt ** 2 * pos).tolist()  # Eq 9.2

        def evolve(idx: int, tuna: Tuna) -> Tuna:
            pos = tuna.position
            prev_pos = pos if idx == 0 else self._population[idx - 1].position
            return Tuna(**self._init_agent(get_new_local_pos(pos, prev_pos)).model_dump())

        aa = 0.7
        zz = 0.05
        epoch = self._current_cycle
        epochs = self._config.max_cycles
        C = epoch / epochs
        a1 = aa + (1 - aa) * C
        a2 = (1 - aa) - (1 - aa) * C
        tt = (1 - C) ** C
        best_position = np.array(self._best_agent.position)
        n_dims = self._task.space_dimension

        self._population = [
            self._greedy_select_agent(tuna, evolve(idx, tuna)) for idx, tuna in enumerate(self._population)
        ]
