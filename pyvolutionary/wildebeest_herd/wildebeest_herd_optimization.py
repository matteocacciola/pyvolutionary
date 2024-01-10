from itertools import chain
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
    def __init__(self, config: WildebeestHerdOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def get_best_local(wb: Wildebeest) -> Wildebeest:
            local_list = []
            pos = np.array(wb.position)
            for _ in range(0, n_explore_step):
                local_list.append(self._init_agent(pos + eta * np.random.uniform() * self._uniform_position()))
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

        def starvation_avoidance(wildebeest: Wildebeest) -> list[Wildebeest]:
            dist_to_worst = distance(wildebeest.position, g_worst.position)
            children = []
            if dist_to_worst < delta_w:
                children.append(
                    Wildebeest(**self._init_agent(self._increase_position(wildebeest.position)).model_dump()))
            return children

        def population_pressure(wildebeest: Wildebeest) -> list[Wildebeest]:
            dist_to_best = distance(wildebeest.position, g_best.position)
            children = []
            if 1.0 < dist_to_best < delta_c:
                children.append(Wildebeest(**self._init_agent(
                    np.array(g_best.position) + self._config.eta * self._uniform_position()
                ).model_dump()))
            return children

        def herd_social_memory() -> list[Wildebeest]:
            return [Wildebeest(
                **self._init_agent(np.array(g_best.position) + 0.1 * self._uniform_position()).model_dump()
            ) for _ in range(0, n_exploit_step)]

        def generate_children(wildebeest: Wildebeest) -> list[Wildebeest]:
            return starvation_avoidance(wildebeest) + population_pressure(wildebeest) + herd_social_memory()

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
