from itertools import chain
from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    distance,
    sort_and_trim,
    special_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Wildebeest, WildebeestHerdOptimizationConfig


class WildebeestHerdOptimization(OptimizationAbstract):
    """
    Implementation of the Wildebeest Herd Optimization algorithm.

    Args:
        config (WildebeestHerdOptimizationConfig): an instance of WildebeestHerdOptimizationConfig class.
            {parse_obj_doc(WildebeestHerdOptimizationConfig)}

    Bibliography
    ----------
    [1] Amali, D. and Dinakaran, M., 2019. Wildebeest herd optimization: a new global optimization algorithm inspired
        by wildebeest herding behaviour. Journal of Intelligent & Fuzzy Systems, 37(6), pp.8063-8076.
    """
    def __init__(self, config: WildebeestHerdOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = WildebeestHerdOptimizationConfig(**parameters)

    def optimization_step(self):
        def get_best_local(wb: Wildebeest) -> Wildebeest:
            pos = np.array(wb.position)
            local_list = [self._init_agent(
                pos + eta * np.random.uniform() * np.array(self._task.empty_solution())
            ) for _ in range(0, n_explore_step)]
            return Wildebeest(**best_agent(local_list).model_dump())

        def local_movement(wildebeest: Wildebeest) -> Wildebeest:
            best_local_position = np.array(get_best_local(wildebeest).position)
            agent = Wildebeest(**self._init_agent(
                local_alpha * best_local_position + local_beta * (np.array(wildebeest.position) - best_local_position)
            ).model_dump())
            return self._greedy_select_agent(wildebeest, agent)

        def herd_instinct(wildebeest: Wildebeest) -> Wildebeest:
            idr = np.random.choice(range(0, pop_size))
            picked_wildebeest = self._population[idr]
            if picked_wildebeest.cost >= wildebeest.cost or np.random.random() >= phi:
                return wildebeest
            agent = Wildebeest(**self._init_agent(
                global_alpha * np.array(wildebeest.position) + global_beta * np.array(picked_wildebeest.position)
            ).model_dump())
            return self._greedy_select_agent(wildebeest, agent)

        def starvation_avoidance(wildebeest: Wildebeest) -> Wildebeest | None:
            dist_to_worst = distance(wildebeest.position, g_worst.position)
            if dist_to_worst < delta_w:
                return Wildebeest(**self._init_agent(self._task.increase_solution(wildebeest.position)).model_dump())
            return None

        def population_pressure(wildebeest: Wildebeest) -> Wildebeest | None:
            dist_to_best = distance(wildebeest.position, g_best.position)
            if 1.0 < dist_to_best < delta_c:
                return Wildebeest(**self._init_agent(
                    np.array(g_best.position) + self._config.eta * self._task.empty_solution()
                ).model_dump())
            return None

        def herd_social_memory() -> list[Wildebeest]:
            return [Wildebeest(
                **self._init_agent(np.array(g_best.position) + 0.1 * np.array(self._task.empty_solution())).model_dump()
            ) for _ in range(0, n_exploit_step)]

        def generate_children(wildebeest: Wildebeest) -> list[Wildebeest]:
            res = herd_social_memory()
            if s := starvation_avoidance(wildebeest):
                res.append(s)
            if p := population_pressure(wildebeest):
                res.append(p)
            return res

        eta = self._config.eta
        delta_c, delta_w = self._config.delta_c, self._config.delta_w
        local_alpha, local_beta = self._config.local_alpha, self._config.local_beta
        n_explore_step, n_exploit_step = self._config.n_explore_step, self._config.n_exploit_step

        pop_size = self._config.population_size
        phi = self._config.phi
        global_alpha, global_beta = self._config.global_alpha, self._config.global_beta

        self._population = [local_movement(wildebeest) for wildebeest in self._population]
        self._population = [herd_instinct(wildebeest) for wildebeest in self._population]
        (g_best, ), (g_worst, ) = special_agents(self._population, n_best=1, n_worst=1)

        children = list(chain.from_iterable([generate_children(agent) for agent in self._population]))
        children = sort_and_trim(children, self._config.population_size)
        self._greedy_select_population(children)
