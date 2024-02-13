from typing import Any
import numpy as np

from ..helpers import (
    special_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import HungerGamesSearchOptimizationConfig, Individual


class HungerGamesSearchOptimization(OptimizationAbstract):
    """
    Implementation of the Hunger Games Search Optimization algorithm.

    Args:
        config (HungerGamesSearchOptimizationConfig): an instance of HungerGamesSearchOptimizationConfig class.
            {parse_obj_doc(HungerGamesSearchOptimizationConfig)}

    Bibliography
    ----------
    [1] Yang, Y., Chen, H., Heidari, A.A. and Gandomi, A.H., 2021. Hunger games search: Visions, conception,
        implementation, deep analysis, perspectives, and towards performance shifts. Expert Systems with Applications,
        177, p.114864.
    """
    def __init__(self, config: HungerGamesSearchOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = HungerGamesSearchOptimizationConfig(**parameters)

    def _init_agent(self, position: list[Any] | np.ndarray | None = None, hunger: float | None = None) -> Individual:
        agent = super()._init_agent(position=position)
        individual = Individual(**agent.model_dump())
        if hunger is not None:
            individual.hunger = hunger
        return individual

    def optimization_step(self):
        def update_hunger_value(individual: Individual) -> Individual:
            # Eq (2.8) and (2.9)
            r = np.random.random()
            space = np.mean(bandwidth)
            H = (individual.cost - best.cost) / (worst.cost - best.cost + self.EPS) * r * 2 * space
            if H < LH:
                H = LH * (1 + r)
            individual.hunger += H
            if best.cost == individual.cost:
                individual.hunger = 0
            return individual
        
        def evolve(individual: Individual) -> Individual:
            pos = np.array(individual.position)
            # Variation control
            x = individual.cost - best.cost
            E = 0.5 if np.abs(x) > 50 else 2 / (np.exp(x) + np.exp(-x))
            # R limits the range of activity, in which the range of R is gradually reduced to 0
            R = 2 * shrink * np.random.random() - shrink  # Eq. (2.3)
            # Calculate the hungry weight of each position
            W1 = 1
            if np.random.random() < PUP:
                W1 = individual.hunger * pop_size / (total_hunger + self.EPS) * np.random.random()     
            W2 = (1 - np.exp(-np.abs(individual.hunger - total_hunger))) * np.random.random() * 2
            # update position of individual Eq. (2.1)
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 < PUP:
                pos_new = pos * (1 + np.random.normal(0, 1))
            elif r2 > E:
                pos_new = W1 * best_pos + R * W2 * np.abs(best_pos - pos)
            else:
                pos_new = W1 * best_pos - R * W2 * np.abs(best_pos - pos)
            return self._greedy_select_agent(self._init_agent(pos_new, individual.hunger), individual)

        cycle = self._current_cycle
        bandwidth = self._task.bandwidth()
        pop_size = self._config.population_size

        (best, ), (worst, ) = special_agents(self._population, n_best=1, n_worst=1)
        best_pos = np.array(best.position)

        PUP, LH = self._config.PUP, self._config.LH

        # Eq. (2.4)
        shrink = 2 * (1 - (cycle + 1) / self._config.max_cycles)
        self._population = [update_hunger_value(individual) for individual in self._population]
        total_hunger = np.sum([individual.hunger for individual in self._population])
        
        self._population = [evolve(individual) for individual in self._population]
