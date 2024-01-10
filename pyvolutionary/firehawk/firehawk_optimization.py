from itertools import chain
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

    def optimization_step(self):
        def get_groups() -> dict[int, list[FireHawk]]:
            result = {}
            prey_pop = self._population[self.__hn:].copy()
            i = 0
            while i < self.__hn <= len(prey_pop):
                b = np.argsort([distance(fh_pop[i].position, prey.position) for prey in prey_pop])
                alfa = np.random.randint(0, len(prey_pop))
                temp = [prey_pop[b[i]] for i in range(alfa)]
                prey_pop = [prey_pop[i] for i in range(len(prey_pop)) if i not in b[:alfa]]
                if not prey_pop:
                    break
                result[i] = temp
                i += 1
            if len(prey_pop) > 0:
                result[list(result.keys())[-1]] += prey_pop
            return result

        def move_near(agent: FireHawk, near: FireHawk) -> FireHawk:
            return (FireHawk(**self._init_agent(
                self._correct_position(np.array(agent.position) + Ir[0] * GB - Ir[1] * np.array(near.position))
            ).model_dump()))

        def move_firehawk_in_group(agent: FireHawk, to: FireHawk, sub: np.ndarray) -> FireHawk:
            Ir = np.random.uniform(0, 1, size=2)
            return (FireHawk(**self._init_agent(
                self._correct_position(np.array(agent.position) + Ir[0] * np.array(to.position) - Ir[1] * sub)
            ).model_dump()))

        def move_group(idx: int, group: list[FireHawk]) -> list[FireHawk]:
            SPl = np.mean([np.array(agent.position) for agent in group], axis=0)
            return (
                [move_firehawk_in_group(agent, fh_pop[np.random.randint(0, self.__hn)], SP) for agent in group] +
                [move_firehawk_in_group(agent, fh_pop[idx], SPl) for agent in group]
            )

        sort_by_cost(self._population)
        fh_pop = self._population[:self.__hn].copy()
        groups = get_groups()

        SP = np.mean([np.array(agent.position) for agent in self._population], axis=0)
        GB = self._best_agent.cost
        Ir = np.random.uniform(0, 1, size=2)

        pop = [move_near(fh, fh_pop[np.random.randint(0, self.__hn)]) for fh in fh_pop] + \
            list(chain.from_iterable([move_group(idx, group) for idx, group in groups.items() if len(group) > 0]))

        self._replace_and_trim_population(pop)
