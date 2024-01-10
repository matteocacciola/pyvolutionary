import numpy as np

from ..helpers import (
    generate_group_population,
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Coyote, CoyotesOptimizationConfig


class CoyotesOptimization(OptimizationAbstract):
    """
    Implementation of the Coyotes Optimization algorithm.

    Args:
        config (CoyotesOptimizationConfig): an instance of CoyotesOptimizationConfig class.
            {parse_obj_doc(CoyotesOptimizationConfig)}

    Bibliography
    ----------
    [1] Pierezan, J. and Coelho, L.D.S., 2018, July. Coyote optimization algorithm: a new metaheuristic
        for global optimization problems. In 2018 IEEE congress on evolutionary computation (CEC) (pp. 1-8). IEEE.
    """
    def __init__(self, config: CoyotesOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__n_packs: int = 0
        self.__ps: float = 0.
        self.__p_leave: float = 0.
        self.__packs: list[list[Coyote]] = []

    def before_initialization(self):
        self.__n_packs = int(self._config.population_size / self._config.num_coyotes)
        self.__ps = 1. / self._task.space_dimension  # Probability of selecting a dimension
        self.__p_leave = 0.005 * (self._config.num_coyotes ** 2)  # Probability of leaving a pack

    def _init_agent(self, position: list[float] | np.ndarray | None = None, age: int | None = None) -> Coyote:
        agent = super()._init_agent(position)
        return Coyote(**agent.model_dump(), age=age if age is not None else 0)

    def optimization_step(self):
        def evolve_coyote(idx: int, coyote: Coyote, pack: list[Coyote], tend: np.ndarray) -> Coyote:
            rc1, rc2 = np.random.choice(list(set(range(0, self._config.num_coyotes)) - {idx}), 2, replace=False)
            # try to update the social condition according to the alpha and the pack tendency (Eq. 12)
            pos_new = np.array(coyote.position) + np.random.random() * (
                np.array(pack[0].position) - np.array(pack[rc1].position)
            ) + np.random.random() * (tend - np.array(pack[rc2].position))
            # keep the coyotes in the search space (optimization problem constraint) and evaluate the new position
            # it means to evaluate the new social condition (Eq. 13) and apply the adaptation (Eq. 14)
            return self._greedy_select_agent(coyote, Coyote(**self._init_agent(pos_new, coyote.age).model_dump()))

        def evolve_pack(pack: list[Coyote]) -> list[Coyote]:
            # get the coyotes that belong to each pack and compute the social tendency of the pack (Eq. 6)
            sort_by_cost(pack)
            tendency = np.mean(np.array([coyote.position for coyote in pack]), axis=0)
            # update social condition of coyotes (Eq. 8)
            pack = [evolve_coyote(idx, coyote, pack, tendency) for idx, coyote in enumerate(pack)]
            # birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            id_parent1, id_parent2 = np.random.choice(list(range(0, num_coyotes)), 2, replace=False)
            # generate the coyote considering intrinsic and extrinsic influence, with eventual noise
            pup_pos = np.array(np.where(
                np.random.random(dim) < (1. - self.__ps) / 2.,
                pack[id_parent1].position,
                pack[id_parent2].position
            )) * np.random.normal(0, 1)
            pup = Coyote(**self._init_agent(pup_pos.tolist()).model_dump())
            # Verify if the pup will survive
            sort_by_cost(pack)
            # find index of element has cost larger than new child: if existing, new child is good
            if pup.cost < pack[-1].cost:
                # replace the worst element by new child, New born child with age = 0
                pack = sorted(pack, key=lambda x: x.age)
                pack[-1] = pup
            return pack

        dim = self._task.space_dimension
        num_coyotes = self._config.num_coyotes

        self.__packs = generate_group_population(self._population, self.__n_packs, self._config.num_coyotes)
        for p in range(0, self.__n_packs):
            self.__packs[p] = evolve_pack(self.__packs[p])

        # a coyote can leave a pack and enter to another pack (Eq. 4)
        if self.__n_packs > 1 and np.random.random() < self.__p_leave:
            id_pack1, id_pack2 = np.random.choice(list(range(0, self.__n_packs)), 2, replace=False)
            id1, id2 = np.random.choice(list(range(0, self._config.num_coyotes)), 2, replace=False)
            self.__packs[id_pack1][id1], self.__packs[id_pack2][id2] = (
                self.__packs[id_pack2][id2], self.__packs[id_pack1][id1]
            )

        # Update coyotes ages
        self._population = [
            self._init_agent(position=coyote.position, age=coyote.age + 1) for pack in self.__packs for coyote in pack
        ]
