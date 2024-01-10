import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Electromagnet, ElectromagneticFieldOptimizationConfig


class ElectromagneticFieldOptimization(OptimizationAbstract):
    """
    Implementation of the Electromagnetic Field Optimization (EFO) algorithm.

    Args:
        config (ElectromagneticFieldOptimizationConfig): an instance of ElectromagneticFieldOptimizationConfig class.
            {parse_obj_doc(ElectromagneticFieldOptimizationConfig)}

    Bibliography
    ----------
    [1] Abedinpourshotorban, H., Shamsuddin, S.M., Beheshti, Z. and Jawawi, D.N., 2016. Electromagnetic field
        optimization: a physics-inspired metaheuristic optimization algorithm. Swarm and Evolutionary Computation, 26,
        pp.8-22.
    """
    def __init__(self, config: ElectromagneticFieldOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__phi = (1 + np.sqrt(5)) / 2  # golden ratio

    def optimization_step(self):
        def evolve(electromagnet: Electromagnet) -> Electromagnet:
            if np.random.random() < ps_rate:
                r_idx1 = np.random.randint(0, p_field)  # top
                r_idx2 = np.random.randint(n_field, pop_size)  # bottom
                r_idx3 = np.random.randint(n_field1, n_field)  # middle

                pos_new = np.array(self._population[r_idx1].position) + phi * np.random.random() * (
                        best_position - np.array(self._population[r_idx3].position)
                ) + np.random.random() * (best_position - np.array(self._population[r_idx2].position))
            else:
                pos_new = self._init_position()
            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if np.random.random() < r_rate:
                pos_new[np.random.randint(0, n_dims)] = self._uniform_coordinates(np.random.randint(0, n_dims))
            # checking whether the generated number is inside boundary or not
            agent = Electromagnet(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(electromagnet, agent)

        pop_size = self._config.population_size
        best_position = np.array(self._best_agent.position)

        n_field = int(pop_size * (1 - self._config.n_field))
        n_field1 = int(pop_size * self._config.p_field + 1)
        p_field = int(pop_size * self._config.p_field)
        ps_rate = self._config.ps_rate

        phi = self.__phi
        r_rate = self._config.r_rate
        n_dims = self._task.space_dimension

        self._population = [evolve(electromagnet) for electromagnet in self._population]
