import numpy as np

from ..helpers import (
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import AquilaOptimizationConfig


class AquilaOptimization(OptimizationAbstract):
    """
    Implementation of the Aquila Optimization algorithm.

    Args:
        config (AquilaOptimizationConfig): an instance of AquilaOptimizationConfig class.
            {parse_obj_doc(AquilaOptimizationConfig)}

    Bibliography
    ----------
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021.
        Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157,
        p.107250.
    """
    def __init__(self, config: AquilaOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __move__(self, idx: int, pos: list[float]) -> np.ndarray:
        """
        Move the agent to a new position. This method is called by the optimization step. It is not intended to be
        called directly.
        :param idx: the index of the agent in the population.
        :param pos: the current position of the agent.
        :return: the new position of the agent.
        :rtype: np.ndarray
        """
        alpha = delta = 0.1
        g1 = 2 * np.random.random() - 1  # Eq. 16
        g2 = 2 * (1 - self._cycles / self._config.max_cycles)  # Eq. 17
        dim_list = np.array(list(range(1, self._task.space_dimension + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = self._cycles ** ((2 * np.random.random() - 1) / (1 - self._config.max_cycles) ** 2)  # Eq.(15)

        best_position = np.array(self._best_agent.position)
        pop_size = self._config.population_size

        pos = np.array(pos)

        x_mean = np.mean(np.array([agent.cost for agent in self._population]), axis=0)
        levy_step = get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)
        if self._cycles <= (2 / 3) * self._config.max_cycles:  # Eq. 3, 4
            jdx = np.random.choice(list(set(range(0, pop_size)) - {idx}))
            return best_position * (1 - self._cycles / self._config.max_cycles) + np.random.random() * (
                    x_mean - best_position
            ) if np.random.random() < 0.5 else best_position * levy_step + (
                    np.array(self._population[jdx].position) + np.random.random() * (y - x)
            )  # Eq. 5

        return alpha * (best_position - x_mean) - (
            np.random.random() * self._random_position() * delta
        ) if np.random.random() < 0.5 else QF * best_position - (
                g2 * pos * np.random.random()
        ) - g2 * levy_step + np.random.random() * g1  # Eq. 13, 14

    def optimization_step(self):
        for idx, aquila in enumerate(self._population):
            agent = self._init_agent(self._correct_position(self.__move__(idx, aquila.position)))
            self._population[idx] = self._greedy_select_agent(agent, aquila)
