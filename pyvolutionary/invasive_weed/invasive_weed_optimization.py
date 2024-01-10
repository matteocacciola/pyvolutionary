from itertools import chain
import numpy as np

from ..helpers import (
    special_agents,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import InvasiveWeed, InvasiveWeedOptimizationConfig


class InvasiveWeedOptimization(OptimizationAbstract):
    """
    Implementation of the Invasive Weed Optimization algorithm.

    Args:
        config (InvasiveWeedOptimizationConfig): an instance of InvasiveWeedOptimizationConfig class.
            {parse_obj_doc(InvasiveWeedOptimizationConfig)}

    Bibliography
    ----------
    [1] Mehrabian, A.R. and Lucas, C., 2006. A novel numerical optimization algorithm inspired from weed colonization.
        Ecological informatics, 1(4), pp.355-366.
    """
    def __init__(self, config: InvasiveWeedOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def evolve(weed: InvasiveWeed) -> list[InvasiveWeed]:
            ratio = (
                np.random.random() if best.cost == worst.cost else (weed.cost - worst.cost) / (best.cost - worst.cost)
            )
            s = min(int(np.ceil(seed_min + (seed_max - seed_min) * ratio)), int(np.sqrt(pop_size)))
            return [InvasiveWeed(**self._init_agent(
                np.array(weed.position) + sigma * np.random.normal(0, 1, n_dims)
            ).model_dump()) for _ in range(0, s)]

        # get best and worst Invasive Weeds
        (best, ), (worst, ) = special_agents(self._population, n_best=1, n_worst=1)

        seed_min, seed_max = self._config.seed
        pop_size = self._config.population_size
        n_dims = self._task.space_dimension

        # update Standard Deviation
        sigma_start, sigma_end = self._config.sigma
        sigma = (
                        (self._config.max_cycles - self._current_cycle) / (self._config.max_cycles - 1)
        ) ** self._config.exponent * (sigma_start - sigma_end) + sigma_end

        pop_new = list(chain.from_iterable(map(evolve, self._population)))

        # update population
        self._extend_and_trim_population(pop_new)
