import numpy as np

from ..helpers import (
    best_agent,
    distance,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Firefly, FireflySwarmOptimizationConfig


class FireflySwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Firefly Swarm Optimization algorithm.

    Args:
        config (FireflySwarmOptimizationConfig): an instance of FireflySwarmOptimizationConfig class.
            {parse_obj_doc(FireflySwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Yang, Xin-She. "Firefly algorithms for multimodal optimization". International symposium on stochastic algorithms.
        Springer, Berlin, Heidelberg, 2009.
    [2] Yang, Xin-She. "Firefly algorithm, stochastic test functions and design optimization." International Journal of
        Bio-Inspired Computation 2.2 (2010): 78-84. https://doi.org/10.1504/IJBIC.2010.032124
    [3] Yang, Xin-She. "Firefly algorithm, Levy flights and global optimization." Research and development in intelligent
        systems XXVI. Springer, London, 2010. 209-218. https://doi.org/10.1007/978-1-84996-153-4_15
    """

    def __init__(self, config: FireflySwarmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __update_alpha__(self):
        """
        Update alpha parameter. This parameter is used to control the randomness of the movement of the fireflies.
        """
        delta = 1.0 - (10.0 ** -4.0 / 0.9) ** (1.0 / self._cycles)
        self._config.alpha *= (1 - delta) * self._config.alpha

    def __update_population__(self):
        """
        Update the population of fireflies. This method is called at each iteration of the algorithm. It updates the
        position of each firefly based on the position of the other fireflies. The fireflies with the best fitness
        values will attract the other fireflies and the fireflies with the worst fitness values will be attracted by the
        other fireflies. The fireflies with the best fitness values will move towards the fireflies with the worst
        fitness values. The fireflies with the worst fitness values will move away from the fireflies with the best
        fitness values.
        """
        alpha = self._config.alpha
        beta_min = self._config.beta_min
        gamma = self._config.gamma

        n_dims = self._task.space_dimension
        pop_size = self._config.population_size
        for idx in range(0, self._config.population_size):
            position = self._population[idx].position
            cost = self._population[idx].cost
            weighted_pos = [np.array(position) + beta_min * np.exp(
                -gamma * distance(position, firefly.position) / np.sqrt(n_dims)
            ) * np.matmul(
                np.array(firefly.position) - np.array(position), np.random.uniform(0, 1, (n_dims, n_dims))
            ) + alpha * np.random.uniform(0, 1, n_dims)
                for firefly in self._population[idx+1:] if firefly.cost < cost]

            new_agents = (
                [self._init_agent(position) for position in weighted_pos] +
                [self._init_agent() for _ in range(0, pop_size - len(weighted_pos) + 1)]
            )
            self._population[idx] = best_agent(new_agents)

    def optimization_step(self):
        # update alpha
        self.__update_alpha__()

        # replace old population
        self.__update_population__()
