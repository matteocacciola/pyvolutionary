import numpy as np

from ..helpers import (
    distance,
    sort_by_cost,
    parse_obj_doc  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import FireHawk, FireHawkOptimizationConfig


class FireHawkOptimization(OptimizationAbstract):
    """
    Implementation of the Fire Hawk Optimization algorithm.

    Args:
        config (FireHawkOptimizationConfig): an instance of FireHawkOptimizationConfig class.
            {parse_obj_doc(FireHawkOptimizationConfig)}

    Bibliography
    ----------
    [1] Azizi, M., Talatahari, S. & Gandomi, A.H. Fire Hawk Optimizer: a novel metaheuristic algorithm. Artif Intell
        Rev 56, 287–363 (2023). https://doi.org/10.1007/s10462-022-10173-w
    """
    def __init__(self, config: FireHawkOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__hn = np.random.randint(1, np.ceil(self._config.population_size / 5))

    def __new_population__(self) -> tuple[dict[int, list[FireHawk]], list[FireHawk]]:
        sort_by_cost(self._population)

        fh_pop = self._population[:self.__hn].copy()

        groups = {}

        prey_pop = self._population[self.__hn:].copy()
        i = 0
        while i < self.__hn <= len(prey_pop):
            b = np.argsort([distance(fh_pop[i].position, prey.position) for prey in prey_pop])
            alfa = np.random.randint(0, len(prey_pop))
            temp = [prey_pop[b[i]] for i in range(alfa)]
            prey_pop = [prey_pop[i] for i in range(len(prey_pop)) if i not in b[:alfa]]
            if not prey_pop:
                break

            groups[i] = temp
            i += 1

        if len(prey_pop) > 0:
            groups[list(groups.keys())[-1]] += prey_pop

        return groups, fh_pop

    def __move_near__(self, agent: FireHawk, near: FireHawk) -> FireHawk:
        GB = self._best_agent.cost
        Ir = np.random.uniform(0, 1, size=2)
        return (FireHawk(**self._init_agent(
            self._correct_position(np.array(agent.position) + Ir[0] * GB - Ir[1] * np.array(near.position))
        ).model_dump()))

    def __move_group__(
        self, idx: int, group: list[FireHawk], SP: np.ndarray, firehawks: list[FireHawk]
    ) -> list[FireHawk]:
        def move(agent: FireHawk, to: FireHawk, sub: np.ndarray) -> FireHawk:
            Ir = np.random.uniform(0, 1, size=2)
            return (FireHawk(**self._init_agent(
                self._correct_position(np.array(agent.position) + Ir[0] * np.array(to.position) - Ir[1] * sub)
            ).model_dump()))

        SPl = np.mean([np.array(agent.position) for agent in group], axis=0)

        return (
            [move(agent, firehawks[np.random.randint(0, self.__hn)], SP) for agent in group] +
            [move(agent, firehawks[idx], SPl) for agent in group]
        )

    def optimization_step(self):
        groups, fh_pop = self.__new_population__()
        SP = np.mean([np.array(agent.position) for agent in self._population], axis=0)

        pop = [self.__move_near__(fh, fh_pop[np.random.randint(0, self.__hn)]) for fh in fh_pop]
        for idx, group in groups.items():
            pop.extend(self.__move_group__(idx, group, SP, fh_pop))

        self._replace_and_trim_population(pop)
