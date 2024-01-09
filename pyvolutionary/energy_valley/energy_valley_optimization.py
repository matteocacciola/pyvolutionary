from itertools import chain
from typing import Final
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Particle, EnergyValleyOptimizationConfig


class EnergyValleyOptimization(OptimizationAbstract):
    """
    Implementation of the Energy Valley Optimization algorithm.

    Args:
        config (EnergyValleyOptimizationConfig): an instance of EnergyValleyOptimizationConfig class.
            {parse_obj_doc(EnergyValleyOptimizationConfig)}

    Bibliography
    ----------
    [1] Azizi, M., Aickelin, U., A. Khorshidi, H., & Baghalzadeh Shishehgarkhaneh, M. (2023). Energy valley optimizer:
        a novel metaheuristic algorithm for global and engineering optimization. Scientific Reports, 13(1), 226.
    """
    EPS: Final[float] = np.finfo(float).eps
    
    def __init__(self, config: EnergyValleyOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __evolve__(
        self,
        idx: int,
        particle: Particle,
        pos_list: list[list[float]],
        cost_list: list[float],
        best_pos: list[float],
        best_cost: float,
        worst_cost: float
    ) -> list[Particle]:
        pos_list = np.array(pos_list)
        fit_list = np.array(cost_list)

        n_dims = self._task.space_dimension
        best_pos = np.array(best_pos)
        pos = np.array(particle.position)

        dis = np.sqrt(np.sum((pos - pos_list) ** 2, axis=1))
        idx_dis_sort = np.argsort(dis)
        CnPtIdx = np.random.choice(list(set(range(2, self._config.population_size)) - {idx}))
        x_team = pos_list[idx_dis_sort[1:CnPtIdx], :]
        x_avg_team = np.mean(x_team, axis=0)
        x_avg_pop = np.mean(pos_list, axis=0)

        eb = np.mean(fit_list)
        sl = (fit_list[idx] - best_cost) / (worst_cost - best_cost + self.EPS)

        pos_new1 = pos.copy()
        pos_new2 = pos.copy()
        if eb < particle.cost:
            if np.random.random() > sl:
                a1_idx, g1_idx = np.random.randint(0, n_dims, size=2)

                a2_idx = np.random.randint(0, n_dims, size=a1_idx)
                pos_new1[a2_idx] = best_pos[a2_idx]

                g2_idx = np.random.randint(0, n_dims, size=g1_idx)
                pos_new2[g2_idx] = x_avg_team[g2_idx]
            else:
                ir = np.random.uniform(0, 1, size=(2, 2))
                jr = np.random.uniform(0, 1, size=(2, n_dims))

                pos_new1 += jr[0] * (ir[0, 0] * best_pos - ir[0, 1] * x_avg_pop) / sl
                pos_new2 += jr[1] * (ir[1, 0] * best_pos - ir[1, 1] * x_avg_team)
            return [
                Particle(**self._init_agent(pos_new1).model_dump()),
                Particle(**self._init_agent(pos_new2).model_dump()),
            ]

        return [
            Particle(**self._init_agent(pos_new1 + np.random.random() * sl * self._uniform_position()).model_dump()),
        ]

    def optimization_step(self):
        pop_new = list(chain.from_iterable(self._solve_mode_process(
            self.__evolve__,
            [agent.position for agent in self._population],
            [agent.cost for agent in self._population],
            self._best_agent.position,
            self._best_agent.cost,
            self._worst_agent.cost,
        )))

        self._extend_and_trim_population(pop_new)
