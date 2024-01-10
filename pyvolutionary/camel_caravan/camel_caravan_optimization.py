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

    def optimization_step(self):
        def walk(camel: Camel) -> Camel:
            supply = camel.supply * (1 - burden_factor * camel.steps / max_cycles)
            endurance = camel.endurance * (1 - temperature / max_temperature) * (1 - camel.steps / max_cycles)
            new_position = camel.position + np.random.uniform(-1, 1) * (1 - (endurance / config_endurance)
            ) * np.exp(1 - supply / config_supply) * (bc_position - np.array(camel.position))
            if self._is_valid_position(new_position):
                return self._init_agent(position=new_position, endurance=endurance, supply=supply)
            return self._init_agent(position=camel.position, endurance=endurance, supply=supply)

        def oasis(camel: Camel, past_cost: float) -> Camel:
            if np.random.random() <= (1 - self._config.visibility) and camel.cost <= past_cost:
                camel.supply = config_supply
                camel.endurance = config_endurance
            return camel

        def life_cycle(camel: Camel, past_cost: float) -> Camel:
            if past_cost <= death_rate * camel.cost:
                return self._init_agent(endurance=config_endurance, supply=config_supply)
            camel.steps += 1
            return camel

        def evolve(camel: Camel) -> Camel:
            c = camel.model_copy()
            past_cost = c.cost
            c = life_cycle(walk(c), past_cost)
            return oasis(c, past_cost)

        death_rate = self._config.death_rate
        burden_factor = self._config.burden_factor
        max_cycles = self._config.max_cycles
        config_endurance = self._config.endurance
        config_supply = self._config.supply
        min_temperature, max_temperature = self._config.temperatures

        temperature = np.random.uniform(min_temperature, max_temperature)
        bc_position = np.array(best_agent(self._population).position)

        self._population = [evolve(camel) for camel in self._population]
