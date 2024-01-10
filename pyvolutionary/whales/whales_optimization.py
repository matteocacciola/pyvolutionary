import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import WhalesOptimizationConfig, Whale


class WhalesOptimization(OptimizationAbstract):
    """
    Implementation of the Whale Optimization algorithm.

    Args:
        config (WhalesOptimizationConfig): an instance of WhalesOptimizationConfig class.
            {parse_obj_doc(WhalesOptimizationConfig)}

    Bibliography
    ----------
    [1] Seyedali Mirjalili, Andrew Lewis, The Whale Optimization Algorithm, Advances in Engineering Software, Volume 95,
        2016, Pages 51-67, ISSN 0965-9978, https://doi.org/10.1016/j.advengsoft.2016.01.008.
    """
    def __init__(self, config: WhalesOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def evolve(whale: Whale) -> Whale:
            size = self._config.population_size

            position = np.array(whale.position)

            r1 = np.random.rand()  # r1 is a random number in [0,1]
            r2 = np.random.rand()  # r2 is a random number in [0,1]

            A = 2 * a * r1 - a  # Eq. (2.3) in the paper
            C = 2 * r2  # Eq. (2.4) in the paper

            l = (a2 - 1) * np.random.rand() + 1  # parameters in Eq. (2.5)

            if np.random.uniform() < 0.5:
                if np.abs(A) >= 1:
                    pos_rand = np.array(self._population[np.random.randint(0, size)].position)
                    D_X_rand = np.abs(C * pos_rand - position)  # Eq. (2.7)
                    position = pos_rand - A * D_X_rand  # Eq. (2.8)
                else:
                    D_Leader = np.abs(C * leader_position - position)  # Eq. (2.1)
                    position = leader_position - A * D_Leader  # Eq. (2.2)
            else:
                distance_to_leader = np.abs(leader_position - position)  # Eq. (2.5)
                position = distance_to_leader * np.exp(l) * np.cos(l * 2 * np.pi) + leader_position

            agent = Whale(**self._init_agent(position).model_dump())
            return self._greedy_select_agent(whale, agent)

        # a decreases linearly from 2 to 0 in Eq. (2.3)
        a = 2 - 2 * self._current_cycle / self._config.max_cycles

        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 - self._current_cycle / self._config.max_cycles

        # Update the Position of search agents
        leader_position = np.array(self._best_agent.position)

        self._population = [evolve(whale) for whale in self._population]
