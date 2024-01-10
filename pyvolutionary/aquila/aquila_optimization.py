import numpy as np

from ..helpers import (
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import AquilaOptimizationConfig, Aquila


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

    def optimization_step(self):
        def move(idx: int, aquila: Aquila) -> Aquila:
            pos = np.array(aquila.position)
            if current_cycle <= (2 / 3) * max_cycles:  # Eq. 3, 4
                jdx = np.random.choice(list(set(range(0, pop_size)) - {idx}))
                pos_new = best_position * (1 - current_cycle / max_cycles) + np.random.random() * (
                    x_mean - best_position
                ) if np.random.random() < 0.5 else best_position * levy_step + (
                    np.array(self._population[jdx].position) + np.random.random() * (y - x)
                )  # Eq. 5
            else:
                pos_new = alpha * (best_position - x_mean) - (
                        np.random.random() * self._random_position() * delta
                ) if np.random.random() < 0.5 else QF * best_position - (
                        g2 * pos * np.random.random()
                ) - g2 * levy_step + np.random.random() * g1  # Eq. 13, 14
            new_agent = Aquila(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(new_agent, aquila)

        current_cycle = self._current_cycle
        max_cycles = self._config.max_cycles
        dims = self._task.space_dimension

        alpha = delta = 0.1
        g1 = 2 * np.random.random() - 1  # Eq. 16
        g2 = 2 * (1 - current_cycle / max_cycles)  # Eq. 17
        dim_list = np.array(list(range(1, dims + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = current_cycle ** ((2 * np.random.random() - 1) / (1 - max_cycles) ** 2)  # Eq.(15)

        best_position = np.array(self._best_agent.position)
        pop_size = self._config.population_size
        x_mean = np.mean(np.array([agent.cost for agent in self._population]), axis=0)
        levy_step = get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)

        self._population = [move(idx, aquila) for idx, aquila in enumerate(self._population)]
