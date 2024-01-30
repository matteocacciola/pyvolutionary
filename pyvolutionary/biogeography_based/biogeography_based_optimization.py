from typing import Any
import numpy as np

from ..helpers import (
    best_agents,
    roulette_wheel_indexes,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import BiogeographyBasedOptimizationConfig, Population


class BiogeographyBasedOptimization(OptimizationAbstract):
    """
    Implementation of the Biogeography Based Optimization algorithm.

    Args:
        config (BiogeographyBasedOptimizationConfig): an instance of BiogeographyBasedOptimizationConfig class.
            {parse_obj_doc(BiogeographyBasedOptimizationConfig)}

    Bibliography
    ----------
    [1] Simon, D., 2008. Biogeography-based optimization. IEEE transactions on evolutionary computation, 12(6),
        pp.702-713.
    """
    def __init__(self, config: BiogeographyBasedOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__mu: np.ndarray | None = None
        self.__mr: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = BiogeographyBasedOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__mu = (
            self._config.population_size + 1 - np.array(range(1, self._config.population_size + 1))
        ) / (self._config.population_size + 1)
        self.__mr = 1 - self.__mu

    def optimization_step(self):
        def evolve(idx: int, population: Population) -> Population:
            idx_selected, = roulette_wheel_indexes(costs)
            # migration step
            condition = np.random.random(n_dims) < mr[idx]
            pos_new = np.where(condition, self._population[idx_selected].position, population.position)
            # Mutation
            pos_new = np.where(np.random.random(n_dims) < p_m, self._task.empty_solution(), pos_new)
            agent_new = Population(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(population, agent_new)

        n_dims = self._task.space_dimension
        mr, mu = self.__mr, self.__mu
        p_m = self._config.p_m

        pop_elites = best_agents(self._population, n_best=self._config.n_elites)

        # pick a position from which to emigrate (roulette wheel selection)
        costs = np.array([agent.cost for agent in self._population])
        costs /= np.sum(costs)

        self._population = [evolve(idx, agent) for idx, agent in enumerate(self._population)]

        self._extend_and_trim_population(pop_elites)
