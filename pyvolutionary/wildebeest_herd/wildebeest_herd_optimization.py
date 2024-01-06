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

    def __get_best_local__(self, wildebeest: Wildebeest) -> Wildebeest:
        local_list = []
        pos = np.array(wildebeest.position)

        eta = self._config.eta
        for _ in range(0, self._config.n_explore_step):
            local_list.append(self._init_agent(pos + eta * np.random.uniform() * self._uniform_position()))

        return Wildebeest(**best_agent(local_list).model_dump())

    def __local_movement__(self):
        """
        Local movement of wildebeest herd optimization algorithm (Milling behaviour)
        """
        local_alpha, local_beta = self._config.local_alpha, self._config.local_beta
        for idx, wildebeest in enumerate(self._population):
            best_local_position = np.array(self.__get_best_local__(wildebeest).position)
            self._population[idx] = self._greedy_select_agent(wildebeest, self._init_agent(
                local_alpha * best_local_position + local_beta * (np.array(wildebeest.position) - best_local_position)
            ))

    def __herd_instinct__(self) -> tuple[Wildebeest, Wildebeest]:
        """
        Herd instinct of wildebeest herd optimization algorithm (Herd behaviour)
        :return: tuple of best and worst wildebeest
        :rtype: tuple[Wildebeest, Wildebeest]
        """
        pop_size = self._config.population_size
        phi = self._config.phi
        global_alpha, global_beta = self._config.global_alpha, self._config.global_beta
        for idx, wildebeest in enumerate(self._population):
            idr = np.random.choice(range(0, pop_size))
            picked_wildebeest = self._population[idr]
            if picked_wildebeest.cost < wildebeest.cost and np.random.random() < phi:
                self._population[idx] = self._greedy_select_agent(wildebeest, self._init_agent(
                    global_alpha * np.array(wildebeest.position) + global_beta * np.array(picked_wildebeest.position)
                ))

        (g_best, ), (g_worst, ) = special_agents(self._population, n_best=1, n_worst=1)

        return g_best, g_worst

    def __starvation_avoidance__(self, wildebeest: Wildebeest, g_worst: Wildebeest) -> tuple[Wildebeest, bool]:
        """
        Starvation avoidance of wildebeest herd optimization algorithm (Starvation behaviour). If the distance between
        wildebeest and worst wildebeest is less than delta_w, then wildebeest will move to another position. The new
        position is generated by adding random number between 0 and 1 to the current position of wildebeest. The new
        position will be clipped to the lower and upper bounds.
        :param wildebeest: the current wildebeest
        :param g_worst: the worst wildebeest
        :return: tuple of wildebeest and boolean value, if wildebeest is moved to another position, then return True,
            otherwise False
        :rtype: tuple[Wildebeest, bool]
        """
        dist_to_worst = distance(wildebeest.position, g_worst.position)
        if dist_to_worst < self._config.delta_w:
            return Wildebeest(
                **self._init_agent(self._increase_position(wildebeest.position)).model_dump()
            ), True

        return wildebeest, False

    def __population_pressure__(self, wildebeest: Wildebeest, g_best: Wildebeest) -> tuple[Wildebeest, bool]:
        """
        Population pressure of wildebeest herd optimization algorithm (Population pressure behaviour). If the distance
        between wildebeest and best wildebeest is less than delta_c, then wildebeest will move to another position. The
        new position is generated by adding random number between 0 and 1 to the current position of wildebeest. The
        new position will be clipped to the lower and upper bounds.
        :param wildebeest: the current wildebeest
        :param g_best: the best wildebeest
        :return: tuple of wildebeest and boolean value, if wildebeest is moved to another position, then return True,
            otherwise False
        :rtype: tuple[Wildebeest, bool]
        """
        dist_to_best = distance(wildebeest.position, g_best.position)

        if 1.0 < dist_to_best < self._config.delta_c:
            return Wildebeest(**self._init_agent(
                np.array(g_best.position) + self._config.eta * self._uniform_position()
            ).model_dump()), True

        return wildebeest, False

    def __herd_social_memory__(self, g_best: Wildebeest) -> list[Wildebeest]:
        wildebeest_list = []
        for jdx in range(0, self._config.n_exploit_step):
            wildebeest_list.append(Wildebeest(
                **self._init_agent(np.array(g_best.position) + 0.1 * self._uniform_position()).model_dump()
            ))

        return wildebeest_list

    def optimization_step(self):
        self.__local_movement__()

        g_best, g_worst = self.__herd_instinct__()

        children = []
        for wildebeest in self._population:
            # starvation avoidance
            new_wildebeest, moved = self.__starvation_avoidance__(wildebeest, g_worst)
            if moved:
                children.append(new_wildebeest)

            # population pressure
            new_wildebeest, moved = self.__population_pressure__(wildebeest, g_best)
            if moved:
                children.append(new_wildebeest)

            # herd social memory
            children.extend(self.__herd_social_memory__(g_best))

        children = sort_and_trim(children, self._config.population_size)
        self._greedy_select_population(children)
