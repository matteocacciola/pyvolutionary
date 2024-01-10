import numpy as np

from ..helpers import (
    distance,
    verser,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Grasshopper, GrasshopperOptimizationConfig


class GrasshopperOptimization(OptimizationAbstract):
    """
    Implementation of the Grasshopper Optimization Algorithm.

    Args:
        config (GrasshopperOptimizationConfig): an instance of GrasshopperOptimizationConfig class.
            {parse_obj_doc(GrasshopperOptimizationConfig)}

    Bibliography
    ----------
    [1] Saremi, S., Mirjalili, S. and Lewis, A., 2017. Grasshopper optimisation algorithm: theory and application.
        Advances in Engineering Software, 105, pp.30-47.
    """
    def __init__(self, config: GrasshopperOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def f(r_vector: np.ndarray):
            return 0.5 * np.exp(-r_vector / 1.5) - np.exp(-r_vector)

        def s_function(position: list[float]):
            """
            Eq.(2.3) in the paper. This function is used to calculate the sum of the attraction forces between the current
            grasshopper and the other grasshoppers in the population. The sum of the attraction forces is calculated by
            multiplying the attraction force between the current grasshopper and each other grasshopper by the direction of
            the attraction force. The direction of the attraction force is calculated by the verser of the vector between
            the current grasshopper and each other grasshopper. The attraction force between the current grasshopper and
            each other grasshopper is calculated by the Eq.(2.2) in the paper. The Eq.(2.2) is a function of the distance
            between the current grasshopper and each other grasshopper. The distance between the current grasshopper and
            each other grasshopper is calculated by the Eq.(2.1) in the paper. The Eq.(2.1) is a function of the position
            of the current grasshopper and the position of each other grasshopper. The position of the current grasshopper
            is the position of the current grasshopper in the current iteration. The position of each other grasshopper is
            the position of each other grasshopper in the current iteration. The Eq.(2.3) is a function of the position of
            the current grasshopper and the position of each other grasshopper. The position of the current grasshopper is
            the position of the current grasshopper in the current iteration. The position of each other grasshopper is the
            position of each other grasshopper in the current iteration.
            :param position: the position of the current grasshopper in the current iteration
            :param ran: a random vector
            :return: the sum of the attraction forces between the current grasshopper and the other grasshoppers in the
            population
            :rtype: np.ndarray
            """
            return np.sum([
                ran * f(2 + np.remainder(distance(position, g.position), 2)) * verser(position, g.position)
                for jdx, g in enumerate(self._population)
            ], axis=0)

        def evolve(grasshopper: Grasshopper) -> Grasshopper:
            sum_grass = s_function(grasshopper.position)
            # Eq. (2.7)
            agent = Grasshopper(**self._init_agent(
                c * np.random.normal(0, 1, self._task.space_dimension) * sum_grass + best_pos
            ).model_dump())
            return self._greedy_select_agent(agent, grasshopper)

        # Eq.(2.8)
        c = self._config.c_max - self._current_cycle * ((self._config.c_max - self._config.c_min) / self._config.max_cycles)
        ran = (c / 2) * self._bandwidth()

        best_pos = np.array(self._best_agent.position)
        self._population = [evolve(grasshopper) for grasshopper in self._population]
