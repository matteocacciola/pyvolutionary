from itertools import chain
import numpy as np

from ..helpers import (
    best_agents,
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Earthworm, EarthwormsOptimizationConfig


class EarthwormsOptimization(OptimizationAbstract):
    """
    Implementation of the Earthworms Optimization algorithm.

    Args:
        config (EarthwormsOptimizationConfig): an instance of EarthwormsOptimizationConfig class.
            {parse_obj_doc(EarthwormsOptimizationConfig)}

    Bibliography
    ----------
    [1] Wang, Gai-Ge & Deb, Suash & Coelho, Leandro. (2015). Earthworm optimization algorithm: a bio-inspired
        metaheuristic algorithm for global optimization problems. International Journal of Bio-Inspired Computation.
        DOI: 10.1504/IJBIC.2015.10004283.
    """
    def __init__(self, config: EarthwormsOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__dyn_beta = self._config.beta

    def optimization_step(self):
        def evolve(idx: int, earthworm: Earthworm) -> Earthworm:
            """
            The earthworm optimization algorithm process. It is composed of two reproduction operators and a mutation
            operator. The reproduction operators are used to generate new individuals, and the mutation operator is used
            to improve the individuals. The reproduction operators are: (1) the first way of reproducing, and (2) the
            second way of reproducing. The mutation operator is: Cauchy's mutation (CM). The reproduction operators and
            the mutation operator are applied to all individuals in the population.
            :param idx: the index of the current individual
            :param earthworm: the current individual
            :return: the evolved individual
            """
            position = np.array(earthworm.position)
            # reproduction 1: the first way of reproducing
            x_t1 = sum_bounds - alpha * np.array(position)
            # reproduction 2: the second way of reproducing
            x_child = self._population[np.random.randint(0, n_chromes - 1)].position
            if idx >= keep:  # select two parents to mate and create two children
                # 80% parents selected from best population
                rng = range(0, int(n_chromes * 0.2)) if np.random.uniform(0, 1) < 0.5 else range(idx, n_chromes)
                idx1, idx2 = np.random.choice(rng, 2, replace=len(rng) < 2)
                r = np.random.rand()
                x_child = (
                    r * np.array(self._population[idx2].position) + (1 - r) * np.array(self._population[idx1].position)
                )
            return Earthworm(**self._init_agent(beta * np.array(x_t1) + (1 - beta) * np.array(x_child)).model_dump())

        def cauchy_mutation() -> Earthworm:
            """
            Cauchy's mutation (CM): make sure each individual is legal
            :return: the Cauchy's mutated individual
            :rtype: Earthworm
            """
            cauchy_w = np.where(np.random.rand() < self._config.prob_mutate, x_mean, best_pos)
            x_t1 = (cauchy_w + best_pos) / 2
            return Earthworm(**self._init_agent(x_t1).model_dump())

        def find_idx_duplicates(chrome: Earthworm) -> list[int]:
            return [jdx for jdx, c in enumerate(self._population[(idx + 1):], idx + 1) if np.array_equal(
                chrome.position, c.position
            )]

        alpha = self._config.alpha
        beta = self.__dyn_beta
        n_chromes = self._config.population_size
        keep = self._config.keep
        dims = self._task.space_dimension

        x_mean = np.mean(np.array([agent.position for agent in self._population]), axis=0)
        best_pos = self._best_agent.position

        sum_bounds = self._sum_bounds()

        # the earthworm optimization algorithm process
        self._population = [evolve(idx, earthworm) for idx, earthworm in enumerate(self._population)]

        # Cauchy's mutation (CM): make sure each individual is legal and doesn't have duplicates
        self._population = self._population[:keep] + (
            [cauchy_mutation() for _ in range(keep, self._config.population_size)]
        )

        # elitism strategy: save the best "keep" earthworms in a temporary array
        # sorted from best, i.e., minor cost, to worst
        chrome_keep = best_agents(self._population, self._config.keep)
        sort_by_cost(self._population)
        for idx in range(0, keep):
            self._population[n_chromes - idx - 1] = chrome_keep[idx].model_copy()

        # clear duplicates in the population
        duplicates = list(chain.from_iterable([find_idx_duplicates(chrome) for chrome in self._population]))
        dimension_to_change = np.random.randint(0, dims - 1, len(duplicates))
        for jdx in duplicates:
            position_jdx = np.array(self._population[jdx].position)
            position_jdx[dimension_to_change[jdx]] = self._uniform_coordinates(dimension_to_change[jdx])
            self._population[jdx] = Earthworm(**self._init_agent(position_jdx).model_dump())

        self.__dyn_beta *= self._config.gamma
