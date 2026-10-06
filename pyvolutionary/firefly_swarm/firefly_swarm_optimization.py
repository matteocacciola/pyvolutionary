from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
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
    [4] Gandomi, A.H., Yang, X.S. and Alavi, A.H., 2011. Mixed variable structural optimization using firefly
        algorithm. Computers & Structures, 89(23-24), pp.2325-2336.

    The implementation follows the one of mealpy (OriginalFFA), with two differences: a firefly is replaced by its best
    candidate only if better than the firefly itself (mealpy compares it with the firefly after its moves), and the
    mutation coefficient is damped at every cycle (mealpy damps the initial value, so it stays constant).
    """

    def __init__(self, config: FireflySwarmOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__alpha: float | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = FireflySwarmOptimizationConfig(**parameters)

    def before_initialization(self):
        # the mutation coefficient is damped during the optimization: keep it in the state of the run, not in the
        # configuration
        self.__alpha = self._config.alpha

    def optimization_step(self):
        def move(firefly: Firefly, brighter: Firefly) -> Firefly:
            """
            Move a firefly towards a brighter one: the attraction decreases with the distance between them, and a random
            step is added.
            :param firefly: the firefly to move
            :param brighter: the brighter firefly
            :return: the moved firefly
            :rtype: Firefly
            """
            position = np.array(firefly.position, dtype=float)
            brighter_position = np.array(brighter.position, dtype=float)
            # radius and attraction level
            rij = np.linalg.norm(position - brighter_position) / d_max
            beta = beta_base * np.exp(-gamma * rij ** exponent)
            # random step
            mutation_vector = delta * np.random.uniform(0, 1, n_dims)
            temp = np.matmul(brighter_position - position, np.random.uniform(0, 1, (n_dims, n_dims)))
            pos_new = position + alpha * mutation_vector + beta * temp
            return Firefly(**self._init_agent(pos_new).__dict__)

        def update_firefly(idx: int) -> Firefly:
            """
            The firefly moves towards each brighter firefly following it in the population, starting each move from
            where the previous one ended. If the moves are fewer than the population size, random fireflies complete
            the candidates. The firefly is replaced by the best candidate, if better.
            :param idx: the index of the firefly
            :return: the updated firefly
            :rtype: Firefly
            """
            firefly = self._population[idx]
            moved = firefly
            candidates = []
            for brighter in self._population[(idx + 1):]:
                if brighter.cost < moved.cost:
                    moved = move(moved, brighter)
                    candidates.append(moved)
            if len(candidates) < pop_size:
                candidates += [Firefly(**agent.__dict__) for agent in self._generate_agents(pop_size - len(candidates))]
            local_best = best_agent(candidates)
            return local_best if local_best.cost < firefly.cost else firefly

        alpha = self.__alpha
        beta_base = self._config.beta_min
        gamma = self._config.gamma
        delta = self._config.delta
        exponent = self._config.exponent

        n_dims = self._task.space_dimension
        d_max = np.sqrt(n_dims)
        pop_size = self._config.population_size

        # the fireflies are updated in place, one after the other (each one is compared with the following ones only,
        # which are not updated yet)
        for idx in range(0, pop_size):
            self._population[idx] = update_firefly(idx)

        # damp the mutation coefficient
        self.__alpha *= self._config.alpha_damp
