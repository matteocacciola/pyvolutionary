import numpy as np

from ..helpers import (
    sort_and_trim,
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Tree, ForestOptimizationAlgorithmConfig


class ForestOptimizationAlgorithm(OptimizationAbstract):
    """
    Implementation of the Forest Optimization Algorithm.

    Args:
        config (ForestOptimizationAlgorithmConfig): an instance of ForestOptimizationAlgorithmConfig class.
            {parse_obj_doc(ForestOptimizationAlgorithmConfig)}

    Bibliography
    ----------
    [1] Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications,
        Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.
    """
    def __init__(self, config: ForestOptimizationAlgorithmConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__dx: np.ndarray | None = None

    def before_initialization(self):
        _, ub = self._get_bounds()
        self.__dx = np.absolute(ub) / 5.0

    def _init_agent(self, position: list[float] | np.ndarray | None = None) -> Tree:
        agent = super()._init_agent(position)
        return Tree(**agent.model_dump())

    def __local_seeding__(self) -> list[Tree]:
        """
        Local optimum search stage that should prevent getting stuck in a local optimum. It is performed by randomly
        changing the position of a tree, i.e. by randomly seeding a tree around it. The number of changes is defined by
        the local_seeding_changes parameter. Finally, the age of each tree is increased by one and the seeded trees are
        added to the population.
        :return: list of seeded trees.
        :rtype: list[Tree]
        """
        def update_position(position: np.ndarray) -> np.ndarray:
            indices = np.random.choice(self._task.space_dimension, self._config.local_seeding_changes, replace=False)
            position[indices] += np.random.uniform(-self.__dx[indices], self.__dx[indices])
            return position

        # repeat local_seeding_changes times each tree that is going to be removed
        candidates = [
            self._init_agent(update_position(np.array(tree.position)))
            for tree in self._population if tree.age == 0
            for _ in range(self._config.local_seeding_changes)
        ]

        # increase the age of each tree
        for tree in self._population:
            tree.age += 1

        return candidates

    def __global_seeding__(self, candidates: list[Tree], size: int) -> list[Tree]:
        """
        Global optimum search stage that should prevent getting stuck in a local optimum. It is performed by randomly
        changing the position of a tree, i.e. by randomly seeding a tree around it. The number of changes is defined by
        the global_seeding_changes parameter. The difference between this method and the local_seeding method is that
        this method is performed on trees that are dying, i.e. that are going to be removed.
        :param candidates: list of trees that are dying, i.e. that are going to be removed.
        :param size: number of seeds to be generated.
        :return: list of seeded trees.
        """
        def update_position(position: np.ndarray) -> np.ndarray:
            indices = np.random.choice(self._task.space_dimension, self._config.global_seeding_changes, replace=False)
            position[indices] = self._uniform_coordinates(indices)
            return position

        return [self._init_agent(
            update_position(np.array(candidates[idx].position))
        ) for idx in np.random.choice(len(candidates), size, replace=False)]

    def __remove_lifetime_exceeded__(self) -> list[Tree]:
        """
        Remove trees that exceeded their lifetime. The lifetime of a tree is defined by the lifetime parameter.
        :return: list of trees that exceeded their lifetime.
        :rtype: list[Tree]
        """
        candidates = [tree for tree in self._population if tree.age > self._config.lifetime]
        # remove trees that exceeded their lifetime from the population
        self._population = [tree for tree in self._population if tree.age <= self._config.lifetime]
        return candidates

    def __survival_of_the_fittest__(self, candidates: list[Tree]):
        """
        Survival of the fittest stage. It is performed by removing the trees with the highest cost function value.
        :param candidates: list of trees that are dying, i.e. that are going to be removed.
        :return: candidates to be removed.
        :rtype: list[Tree]
        """
        sort_by_cost(self._population)
        candidates += self._population[self._config.area_limit+1:]
        self._population = self._population[:self._config.area_limit]
        return candidates

    def optimization_step(self):
        # add seeded trees to the population
        self._population += self.__local_seeding__()

        # identify trees that exceeded their lifetime and area limit
        dying_trees = self.__survival_of_the_fittest__(
            self.__remove_lifetime_exceeded__()
        )

        # global seeding
        gsn = int(self._config.transfer_rate * len(dying_trees))
        if gsn > 0:
            self._population += self.__global_seeding__(dying_trees, gsn)

        self._population = sort_and_trim(self._population, self._config.population_size)
        self._population[0].age = 0
