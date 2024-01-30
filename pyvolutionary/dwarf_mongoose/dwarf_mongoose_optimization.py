from typing import Any
import numpy as np

from ..helpers import (
    roulette_wheel_indexes,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import DwarfMongoose, DwarfMongooseOptimizationConfig


class DwarfMongooseOptimization(OptimizationAbstract):
    """
    Implementation of the Dwarf Mongoose Optimization algorithm.

    Args:
        config (DwarfMongooseOptimizationConfig): an instance of DwarfMongooseOptimizationConfig class.
            {parse_obj_doc(DwarfMongooseOptimizationConfig)}

    Bibliography
    ----------
    [1] Agushaka, J. O., Ezugwu, A. E., & Abualigah, L. (2022). Dwarf mongoose optimization algorithm. Computer methods
        in applied mechanics and engineering, 391, 114570.
    """
    def __init__(self, config: DwarfMongooseOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = DwarfMongooseOptimizationConfig(**parameters)

    def _init_agent(self, position: list[float] | np.ndarray | None = None) -> DwarfMongoose:
        return DwarfMongoose(**super()._init_agent(position).model_dump())

    def optimization_step(self):
        def foraging(idx: int, agent: DwarfMongoose) -> DwarfMongoose:
            agent.C += 1
            alpha, = roulette_wheel_indexes(fi)
            k = np.random.choice(list(set(range(0, pop_size)) - {idx, alpha}))
            # define vocalization coefficient
            phi = (peep / 2) * np.random.uniform(-1, 1, n_dims)
            pos_alpha = np.array(self._population[alpha].position)
            new_pos = pos_alpha + phi * (pos_alpha - np.array(self._population[k].position))
            return self._greedy_select_agent(agent, self._init_agent(new_pos))
        
        def scouting(idx: int, agent: DwarfMongoose) -> tuple[DwarfMongoose, float]:
            agent.C += 1
            pos = np.array(agent.position)
            k = np.random.choice(list(set(range(0, pop_size)) - {idx}))
            # define vocalization coefficient
            phi = (peep / 2) * np.random.uniform(-1, 1, n_dims)
            new_pos = pos + phi * (pos - np.array(self._population[k].position))
            new_agent = self._init_agent(new_pos)
            # sleeping mould
            SM = (new_agent.cost - agent.cost) / (np.max([new_agent.cost, agent.cost]) + self.EPS)
            return self._greedy_select_agent(agent, new_agent), SM

        def babysitting(idx: int, agent: DwarfMongoose) -> DwarfMongoose:
            if agent.C >= L and idx < n_baby_sitter:
                return self._init_agent()
            return agent

        def evolve(idx: int, agent: DwarfMongoose) -> DwarfMongoose:
            pos = np.array(agent.position)
            phi = (peep / 2) * np.random.uniform(-1, 1, n_dims)
            factor = np.array(self._best_agent.position) - SM_list[idx] * pos
            new_pos = pos + CF * phi * factor
            if new_tau > SM_list[idx]:
                new_pos = np.array(self._best_agent.position) - CF * phi * factor
            return self._greedy_select_agent(self._init_agent(new_pos), agent)

        epoch = self._current_cycle
        epochs = self._config.max_cycles

        cost_list = np.array([agent.cost for agent in self._population])
        fi = np.exp(-cost_list / (np.mean(cost_list) + self.EPS))

        pop_size = self._config.population_size
        n_dims = self._task.space_dimension
        peep = self._config.peep
        n_baby_sitter = self._config.n_baby_sitter

        # foraging
        self._population = [foraging(idx, agent) for idx, agent in enumerate(self._population)]

        # scouting
        self._population, SM_list = map(
            lambda x: list(x), zip(*[scouting(idx, agent) for idx, agent in enumerate(self._population)])
        )

        # babysitting
        L = np.round(0.6 * n_dims * n_baby_sitter)
        self._population = [babysitting(idx, agent) for idx, agent in enumerate(self._population)]

        # next mongoose position
        CF = (1. - epoch / epochs) ** (2. * epoch / epochs)  # abandonment counter
        new_tau = np.mean(SM_list)
        self._population = [evolve(idx, agent) for idx, agent in enumerate(self._population)]
