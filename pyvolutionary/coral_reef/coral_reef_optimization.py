import numpy as np

from ..helpers import (
    sort_and_trim,
    parse_obj_doc  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Coral, CoralReefOptimizationConfig


class CoralReefOptimization(OptimizationAbstract):
    """
    Implementation of the Coral Reef Optimization algorithm.

    Args:
        config (CoralReefOptimizationConfig): an instance of CoralReefOptimizationConfig class.
            {parse_obj_doc(CoralReefOptimizationConfig)}

    Bibliography
    ----------
    [1] Salcedo-Sanz, S., Del Ser, J., Landa-Torres, I., Gil-López, S. and Portilla-Figueras, J.A., 2014.
        The coral reefs optimization algorithm: a novel metaheuristic for efficiently solving optimization problems.
        The Scientific World Journal, 2014.
    """
    def __init__(self, config: CoralReefOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        _, self.__G1 = self._config.gamma
        self.__alpha = 10 * self._config.Pd / self._config.max_cycles
        self.__gamma = 10 * (self._config.gamma[1] - self._config.gamma[0]) / self._config.max_cycles
        self.__num_occupied = int(self._config.population_size / (1 + self._config.po))
        self.__dyn_Pd = 0
        self.__occupied_list = np.zeros(self._config.population_size, dtype=int)
        self.__occupied_idx_list = np.random.choice(
            list(range(self._config.population_size)), self.__num_occupied, replace=False
        )
        self.__occupied_list[self.__occupied_idx_list] = 1
        self.__reset_count = 0

    def __broadcast_spawning_brooding__(self) -> list[Coral]:
        """
        Broadcast spawning and brooding process. This process is done in two steps: 1a and 1b. In step 1a, the corals
        that are going to broadcast spawning are selected. In step 1b, the corals that are going to brooding are
        selected. The corals that are going to broadcast spawning are selected randomly from the occupied corals. The
        corals that are going to brooding are selected randomly from the occupied corals that are not selected for
        broadcast spawning.
        :return: the list of larvae
        :rtype: list[Coral]
        """
        def gaussian_mutation(position) -> list[float]:
            return self._correct_position(np.where(
                np.random.random(self._task.space_dimension) < self._config.GCR,
                position + self.__G1 * self._bandwidth() * np.random.normal(0, 1,  self._task.space_dimension),
                position
            ))

        def multi_point_cross(pos1, pos2) -> list[float]:
            p1, p2 = np.random.choice(list(range(0, len(pos1))), 2, replace=False)
            start, end = min(p1, p2), max(p1, p2)
            return self._correct_position(
                np.concatenate((pos1[:start], pos2[start:end], pos1[end:]), axis=0)
            )

        # Step 1a
        selected_corals = np.random.choice(
            self.__occupied_idx_list, int(len(self.__occupied_idx_list) * self._config.Fb), replace=False
        )
        larvae = [Coral(**self._init_agent(gaussian_mutation(self._population[idx].position)).model_dump())
                  for idx in self.__occupied_idx_list if idx not in selected_corals]

        # Step 1b
        while len(selected_corals) >= 2:
            id1, id2 = np.random.choice(range(len(selected_corals)), 2, replace=False)
            agent = Coral(**self._init_agent(multi_point_cross(
                self._population[selected_corals[id1]].position,
                self._population[selected_corals[id2]].position
            )).model_dump())
            larvae.append(agent)
            selected_corals = np.delete(selected_corals, [id1, id2])
        return larvae

    def __larvae_setting__(self, larvae: list[Coral]) -> None:
        """
        Larvae setting process. This process is done in two steps: 2a and 2b. In step 2a, the larvae are evaluated and
        the larvae that are going to be settled are selected. In step 2b, the larvae that are going to be settled are
        placed in the occupied corals.
        :param larvae: the list of larvae
        """
        def find_available_slot(lc: Coral) -> tuple[int | None, bool]:
            for i in range(self._config.n_trials):
                p = np.random.randint(0, self._config.population_size - 1)
                if self.__occupied_list[p] == 0:
                    return p, True
                if lc.cost < self._population[p].cost:
                    return p, False
            return None, False

        def update(p: int, upd: bool, lc: Coral):
            self._population[p] = lc  # update population
            if upd:
                self.__occupied_idx_list = np.append(self.__occupied_idx_list, p)  # update occupied id
                self.__occupied_list[p] = 1  # update occupied list

        # trial to land on a square of reefs
        res = list(map(find_available_slot, larvae))
        for idx, (pdx, upd_occupied) in enumerate(res):
            if pdx is not None:
                update(pdx, upd_occupied, larvae[idx])

    def __reef_cost__(self, idx) -> float:
        return self._population[idx].cost

    def optimization_step(self):
        # broadcast spawning brooding
        larvae = self.__broadcast_spawning_brooding__()
        self.__larvae_setting__(larvae)

        # asexual reproduction
        pop_best = sort_and_trim(
            [self._population[idx] for idx in self.__occupied_idx_list],
            int(len(self.__occupied_idx_list) * self._config.Fa)
        )
        self.__larvae_setting__(pop_best)

        # depredation
        if np.random.random() < self.__dyn_Pd:
            num_depredation = int(len(self.__occupied_idx_list) * self._config.Fd)
            idx_list_sorted = sorted(self.__occupied_idx_list, key=self.__reef_cost__)
            for idx in idx_list_sorted[-num_depredation:]:
                self.__occupied_list[idx] = 0
        if self.__dyn_Pd <= self._config.Pd:
            self.__dyn_Pd += self.__alpha
        if self.__G1 >= self._config.gamma[0]:
            self.__G1 -= self.__gamma
