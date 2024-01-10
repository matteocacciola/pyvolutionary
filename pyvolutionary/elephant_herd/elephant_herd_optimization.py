import numpy as np

from ..helpers import (
    generate_group_population,
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Elephant, ElephantHerdOptimizationConfig


class ElephantHerdOptimization(OptimizationAbstract):
    """
    Implementation of the Elephant Herd Optimization algorithm.

    Args:
        config (ElephantHerdOptimizationConfig): an instance of ElephantHerdOptimizationConfig class.
            {parse_obj_doc(ElephantHerdOptimizationConfig)}

    Bibliography
    ----------
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2015, December. Elephant herding optimization.
        In 2015 3rd international symposium on computational and business intelligence (ISCBI) (pp. 1-5). IEEE.
    """
    def __init__(self, config: ElephantHerdOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__n_individuals: int = int(self._config.population_size / self._config.n_clans)
        self.__groups: list[list[Elephant]] = []

    def _init_population(self):
        super()._init_population()
        self.__groups = generate_group_population(self._population, self._config.n_clans, self.__n_individuals)

    def optimization_step(self):
        def evolve(idx: int, elephant: Elephant) -> Elephant:
            clan_idx = int(idx / n_individuals)
            pos_group = [np.array(elephant.position) for elephant in self.__groups[clan_idx]]
            # pos_clan_idx == 0 means the best in clan, because all clans are sorted based on cost
            pos_clan_idx = int(idx % n_individuals)
            pos_new = beta * np.mean(pos_group, axis=0) if pos_clan_idx == 0 else (
                pos_group[pos_clan_idx] + alpha * np.random.random() * (pos_group[0] - pos_group[pos_clan_idx])
            )
            agent = Elephant(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(elephant, agent)

        n_individuals = self.__n_individuals
        alpha = self._config.alpha
        beta = self._config.beta
        self._population = [evolve(idx, elephant) for idx, elephant in enumerate(self._population)]
        self.__groups = generate_group_population(self._population, self._config.n_clans, self.__n_individuals)

        # Separating operator
        for idx in range(0, self._config.n_clans):
            sort_by_cost(self.__groups[idx])
            self.__groups[idx][-1] = self._init_agent()
        self._population = [elephant for pack in self.__groups for elephant in pack]
