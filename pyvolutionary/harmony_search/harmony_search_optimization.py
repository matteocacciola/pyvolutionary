import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Harmony, HarmonySearchOptimizationConfig


class HarmonySearchOptimization(OptimizationAbstract):
    """
    Implementation of the Harmony Search Optimization algorithm.

    Args:
        config (HarmonySearchOptimizationConfig): an instance of HarmonySearchOptimizationConfig class.
            {parse_obj_doc(HarmonySearchOptimizationConfig)}

    Bibliography
    ----------
    [1] Zong Woo Geem, Joong Hoon Kim, Loganathan GV. A New Heuristic Optimization Algorithm: Harmony Search.
        SIMULATION. 2001;76(2):60-68. doi:10.1177/003754970107600201
    """
    def __init__(self, config: HarmonySearchOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__fw: np.ndarray | None = None
        self.__dyn_fw: np.ndarray | None = None
        self.__fw_damp = 0.9995  # Damp Ratio

    def before_initialization(self):
        self.__fw = 0.0001 * self._bandwidth()  # Bandwidth
        self.__dyn_fw = self.__fw

    def optimization_step(self):
        def create_new_harmony() -> Harmony:
            """
            Create a new harmony. This method is used in the optimization step. It creates a new harmony based on the best
            solution and the harmony memory. The new harmony is created using the following steps: (1) create a new harmony
            position, (2) use the harmony memory, (3) pitch adjustment. The pitch adjustment is a random process that can be
            used to improve the new harmony position.
            :return: a new harmony
            :rtype: Harmony
            """
            # Create New Harmony Position
            pos_new = self._uniform_position()

            # Use Harmony Memory
            pos_new = np.where(np.random.random(dim) < consideration_rate, best_pos, pos_new)

            # Pitch Adjustment
            pos_new = np.where(np.random.random(dim) < pitch_adjusting_rate, pos_new + delta, pos_new)

            # Create New Harmony
            return Harmony(**self._init_agent(pos_new).model_dump())

        delta = self.__dyn_fw * self._uniform_position()
        best_pos = self._best_agent.position
        consideration_rate = self._config.consideration_rate
        pitch_adjusting_rate = self._config.pitch_adjusting_rate

        dim = self._task.space_dimension

        pop_new = [create_new_harmony() for _ in range(self._config.population_size)]

        self.__dyn_fw *= self.__fw_damp
        self._extend_and_trim_population(pop_new)
