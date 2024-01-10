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

    def optimization_step(self):
        def local_seeding(tree: Tree) -> Tree:
            position = np.array(tree.position)
            indices = np.random.choice(dims, local_seeding_changes, replace=False)
            position[indices] += np.random.uniform(-self.__dx[indices], self.__dx[indices])
            return self._init_agent(position)

        def global_seeding(tree: Tree) -> Tree:
            position = np.array(tree.position)
            indices = np.random.choice(self._task.space_dimension, self._config.global_seeding_changes, replace=False)
            position[indices] = self._uniform_coordinates(indices)
            return self._init_agent(position)

        dims, local_seeding_changes = self._task.space_dimension, self._config.local_seeding_changes
        area_limit = self._config.area_limit
        transfer_rate = self._config.transfer_rate

        # add seeded trees to the population
        self._population += [local_seeding(tree) for tree in self._population if tree.age == 0
                             for _ in range(local_seeding_changes)]

        # remove trees that exceeded their lifetime from the population
        dying_trees = [tree for tree in self._population if tree.age > self._config.lifetime]
        self._population = [tree for tree in self._population if tree.age <= self._config.lifetime]

        # identify and remove trees that exceeded their area limit
        sort_by_cost(self._population)
        dying_trees += self._population[area_limit + 1:]

        self._population = self._population[:area_limit]

        # global seeding
        gsn = int(transfer_rate * len(dying_trees))
        if gsn > 0:
            self._population += [
                global_seeding(dying_trees[idx]) for idx in np.random.choice(len(dying_trees), gsn, replace=False)
            ]

        self._population = sort_and_trim(self._population, self._config.population_size)
        self._population = [
            t.model_copy(update={"age": t.age + 1}) if idx > 0 else t for idx, t in enumerate(self._population)
        ]
