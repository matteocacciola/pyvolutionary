from typing import Any
import numpy as np

from ..helpers import (
    split_in_clusters,
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
    def __init__(self, config: ElephantHerdOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__groups: list[list[Elephant]] = []

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = ElephantHerdOptimizationConfig(**parameters)

    def after_initialization(self):
        # every elephant belongs to a clan, also when the population size is not a multiple of n_clans
        self.__groups = split_in_clusters(self._population, self._config.n_clans)

    def optimization_step(self):
        def evolve(idx: int, elephant: Elephant) -> Elephant:
            clan_idx, pos_clan_idx = locations[idx]
            pos_group = [np.array(elephant.position) for elephant in self.__groups[clan_idx]]
            # pos_clan_idx == 0 means the best in clan, because all clans are sorted based on cost
            pos_new = beta * np.mean(pos_group, axis=0) if pos_clan_idx == 0 else (
                pos_group[pos_clan_idx] + alpha * np.random.random() * (pos_group[0] - pos_group[pos_clan_idx])
            )
            agent = Elephant(**self._init_agent(pos_new).__dict__)
            return self._greedy_select_agent(elephant, agent)

        # the (clan, position in the clan) of each elephant, clan by clan
        locations = [(c_id, l_id) for c_id, clan in enumerate(self.__groups) for l_id in range(0, len(clan))]
        alpha = self._config.alpha
        beta = self._config.beta
        self._population = [evolve(idx, elephant) for idx, elephant in enumerate(self._population)]
        self.__groups = split_in_clusters(self._population, self._config.n_clans)

        # Separating operator
        for idx in range(0, len(self.__groups)):
            self.__groups[idx] = sort_by_cost(self.__groups[idx])
            self.__groups[idx][-1] = self._init_agent()
        self._population = [elephant for pack in self.__groups for elephant in pack]
