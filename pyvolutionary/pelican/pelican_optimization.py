import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import PelicanOptimizationConfig


class PelicanOptimization(OptimizationAbstract):
    """
    Implementation of the Pelican Optimization algorithm.

    Args:
        config (PelicanOptimizationConfig): an instance of PelicanOptimizationConfig class.
            {parse_obj_doc(OspreyOptimizationConfig)}

    Bibliography
    ----------
    [1] Trojovský, P., & Dehghani, M. (2022). Pelican optimization algorithm: A novel nature-inspired algorithm for
        engineering applications. Sensors, 22(3), 855.
    """
    def __init__(self, config: PelicanOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        n_dims = self._task.space_dimension
        for idx, pelican in enumerate(self._population):
            # update location of food
            kk = np.random.permutation(self._config.population_size)[0]
            pos_kk = np.array(self._population[kk].position)

            pos = np.array(pelican.position)

            # phase 1: moving towards prey (exploration phase), Eq. 4
            pos_new = pos + np.random.random() * (
                pos_kk - np.random.randint(1, 3) * pos
            ) if self._population[kk].cost < pelican.cost else pos + np.random.random() * (pos - pos_kk)
            agent = self._init_agent(self._correct_position(pos_new))

            pelican = self._greedy_select_agent(pelican, agent)

            # phase 2: Winging on the water surface (exploitation phase), with Eq. 6
            pos_new = np.array(pelican.position) + 0.2 * (1 - self._cycles / self._config.max_cycles) * (
                2 * np.random.random(n_dims) - 1
            ) * np.array(pelican.position)
            agent = self._init_agent(self._correct_position(pos_new))
            self._population[idx] = self._greedy_select_agent(pelican, agent)
