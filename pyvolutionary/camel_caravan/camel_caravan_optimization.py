import numpy as np

from ..helpers import (
    best_agent,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Camel, CamelCaravanOptimizationConfig


class CamelCaravanOptimization(OptimizationAbstract):
    """
    Implementation of the Camels Optimization algorithm.

    Args:
        config (CamelCaravanOptimizationConfig): an instance of CamelCaravanOptimizationConfig class.
            {parse_obj_doc(CamelCaravanOptimizationConfig)}

    Bibliography
    ----------
    [1] Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior.
        Iraq J. Electrical and Electronic Engineering. 12. 167-177.
    """
    def __init__(self, config: CamelCaravanOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def _init_agent(
        self,
        position: list[float] | np.ndarray | None = None,
        endurance: float | None = None,
        supply: float | None = None,
    ) -> Camel:
        agent = super()._init_agent(position=position)

        return Camel(
            **agent.model_dump(),
            endurance=endurance if endurance is not None else self._config.endurance,
            supply=supply if supply is not None else self._config.supply,
            temperature=np.random.uniform(*self._config.temperatures),
        )

    def __walk__(self, camel: Camel) -> Camel:
        """
        Move the camel in search space. This function is used to move Camel.
        :param camel: the camel to consider.
        :return: the moved camel.
        :rtype: Camel
        """
        bc_position = np.array(best_agent(self._population).position)

        min_temperature, max_temperature = self._config.temperatures

        temperature = np.random.uniform(min_temperature, max_temperature)
        supply = camel.supply * (1 - self._config.burden_factor * camel.steps / self._config.max_cycles)
        endurance = camel.endurance * (1 - temperature / max_temperature) * (1 - camel.steps / self._config.max_cycles)

        new_position = camel.position + np.random.uniform(-1, 1) * (
            1 - (endurance / self._config.endurance)
        ) * np.exp(1 - supply / self._config.supply) * (bc_position - np.array(camel.position))
        if self._is_valid_position(new_position):
            return self._init_agent(position=new_position, endurance=endurance, supply=supply)

        return self._init_agent(position=camel.position, endurance=endurance, supply=supply)

    def __oasis__(self, camel: Camel, past_cost: float) -> Camel:
        """
        Apply oasis function to Camel. This function is used to refill supply and endurance of Camel.
        :param camel: the camel to consider.
        :param past_cost: the past cost of Camel.
        :return: the camel after oasis function.
        """
        if np.random.random() <= (1 - self._config.visibility) and camel.cost <= past_cost:
            camel.supply = self._config.supply
            camel.endurance = self._config.endurance
        return camel

    def __life_cycle__(self, camel: Camel, past_cost: float) -> Camel:
        """
        Apply life cycle to Camel. This function is used to kill Camel.
        :param camel: the camel to consider.
        :param past_cost: the past cost of Camel.
        :return: the camel after life cycle.
        """
        if past_cost <= self._config.death_rate * camel.cost:
            return self._init_agent(endurance=self._config.endurance, supply=self._config.supply)

        camel.steps += 1
        return camel

    def optimization_step(self):
        for idx in range(0, self._config.population_size):
            camel = self._population[idx].model_copy()
            past_cost = camel.cost

            camel = self.__life_cycle__(self.__walk__(camel), past_cost)
            self._population[idx] = self.__oasis__(camel, past_cost)
