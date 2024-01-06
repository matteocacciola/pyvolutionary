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
        self.__chrome_keep: list[Earthworm] = []
        self.__dyn_beta = self._config.beta

    def __evolve__(self):
        """
        The earthworm optimization algorithm process. It is composed of two reproduction operators and a mutation
        operator. The reproduction operators are used to generate new individuals, and the mutation operator is used
        to improve the individuals. The reproduction operators are: (1) the first way of reproducing, and (2) the
        second way of reproducing. The mutation operator is: Cauchy's mutation (CM). The reproduction operators and
        the mutation operator are applied to all individuals in the population.
        :return:
        """
        alpha = self._config.alpha
        beta = self.__dyn_beta

        n_chromes = self._config.population_size
        keep = self._config.keep
        for idx in range(0, self._config.population_size):
            position = np.array(self._population[idx].position)

            # reproduction 1: the first way of reproducing
            x_t1 = self._sum_bounds() - alpha * np.array(position)

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

            self._population[idx] = self._init_agent(beta * np.array(x_t1) + (1 - beta) * np.array(x_child))
        self.__dyn_beta *= self._config.gamma

    def __cauchy_mutation__(self):
        """
        Cauchy's mutation (CM): make sure each individual is legal
        """
        keep = self._config.keep

        pos_list = np.array([agent.position for agent in self._population])
        x_mean = np.mean(pos_list, axis=0)
        cauchy_w = self._best_agent.position
        for idx in range(keep, self._config.population_size):
            # Cauchy's mutation (CM)
            cauchy_w = np.where(np.random.rand() < self._config.prob_mutate, x_mean, cauchy_w)
            x_t1 = (cauchy_w + self._best_agent.position) / 2
            # make sure each individual is legal
            self._population[idx] = self._init_agent(x_t1)

    def __clear_duplicates__(self):
        """
        Clear duplicates in the population
        """
        n_chromes = self._config.population_size
        for idx in range(0, n_chromes):
            pos_idx = self._population[idx].position

            jdx_list = [jdx for jdx, chrome in enumerate(self._population[(idx + 1):], idx + 1) if np.array_equal(
                pos_idx, chrome.position
            )]
            for jdx in jdx_list:
                dimension_to_change = np.random.randint(0, self._task.space_dimension - 1)
                position_jdx = self._population[jdx].position
                position_jdx[dimension_to_change] = self._uniform_coordinates(dimension_to_change)
                self._population[jdx] = self._init_agent(position_jdx)

    def __elitism__(self, agents: list[Earthworm], keep: int):
        """
        Elitism operator. Replace the worst individuals with the best ones from the previous cycle.
        :param agents: the agents
        :param keep: the number of best individuals to keep
        """
        # sorted from best, i.e., minor cost, to worst
        sort_by_cost(self._population)
        population_size = self._config.population_size
        for idx in range(0, keep):
            self._population[population_size - idx - 1] = agents[idx].model_copy()

    def optimization_step(self):
        # elitism strategy: save the best "keep" earthworms into the self.__chrome_keep
        self.__chrome_keep = best_agents(self._population, self._config.keep)

        # the earthworm optimization algorithm process
        self.__evolve__()

        # Cauchy's mutation (CM): make sure each individual is legal and doesn't have duplicates
        self.__cauchy_mutation__()

        # elitism strategy: save the best earthworms in a temporary array
        self.__elitism__(self.__chrome_keep, self._config.keep)

        # clear duplicates in the population
        self.__clear_duplicates__()
