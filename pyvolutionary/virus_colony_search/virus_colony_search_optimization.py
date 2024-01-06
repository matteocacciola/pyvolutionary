import numpy as np

from ..helpers import (
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import VirusColonySearchOptimizationConfig


class VirusColonySearchOptimization(OptimizationAbstract):
    """
    Implementation of the Virus Colony Search Optimization algorithm.

    Args:
        config (VirusColonySearchOptimizationConfig): an instance of VirusColonySearchOptimizationConfig class.
            {parse_obj_doc(VirusColonySearchOptimizationConfig)}

    Bibliography
    ----------
    [1] Li, M.D., Zhao, H., Weng, X.W. and Han, T., 2016. A novel nature-inspired algorithm for optimization: Virus
        colony search. Advances in Engineering Software, 92, pp.65-88.
    """
    def __init__(self, config: VirusColonySearchOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__n_best = int(self._config.lamda * self._config.population_size)

    def __viruses_diffusion__(self):
        """
        Viruses diffusion step.
        """
        max_cycles = self._config.max_cycles
        cycle = self._cycles
        best_position = np.array(self._best_agent.position)

        for idx in range(0, self._config.population_size):
            position = np.array(self._population[idx].position)
            sigma = (np.log1p(cycle) / max_cycles) * (position - best_position)
            gauss = np.random.normal(np.random.normal(best_position, np.abs(sigma)))
            position_new = gauss + np.random.uniform() * best_position - np.random.uniform() * position

            self._population[idx] = self._greedy_select_agent(self._population[idx], self._init_agent(position_new))

    def __calculate_x_mean__(self, n_best: int) -> list[float]:
        """
        Calculate the mean position of list of solutions (population). Specifically, it calculates the weighted mean of
        the λ best individuals by using the weights w_i = log(λ + 1) / i * log(λ + 1) / λ * log(λ + 1) / (λ + 1 - i).
        :param n_best: the number of best individuals to consider in the calculation of the mean position (λ)
        :return: the mean position
        :rtype: list[float]
        """
        pop = self._population.copy()
        sort_by_cost(pop)
        positions = [np.array(agent.position) for agent in pop[:n_best]]
        factor_down = n_best * np.log1p(n_best + 1) - np.log1p(np.prod(range(1, n_best + 1)))
        weight = np.log1p(n_best + 1) / factor_down
        weight = weight / n_best
        x_mean = weight * np.sum(positions, axis=0)
        return x_mean

    def __host_cell_infection__(self):
        """
        Host cell infection step.
        """
        x_mean = self.__calculate_x_mean__(self.__n_best)
        sigma = self._config.sigma * (1 - self._cycles / self._config.max_cycles)
        for idx in range(0, self._config.population_size):
            self._population[idx] = self._greedy_select_agent(
                self._population[idx],
                self._init_agent(x_mean + sigma * np.random.normal(0, 1, self._task.space_dimension))
            )

    def __immune_response__(self):
        """
        Immune response step.
        """
        n_dims = self._task.space_dimension
        pop_size = self._config.population_size
        for idx in range(0, self._config.population_size):
            position = np.array(self._population[idx].position)

            pr = (n_dims - idx + 1) / n_dims
            id1, id2 = np.random.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)

            temp = np.array(self._population[id1].position) - np.random.uniform() * (
                np.array(self._population[id2].position) - position
            )
            position_new = np.where(np.random.random(n_dims) < pr, position, temp)
            self._population[idx] = self._greedy_select_agent(self._population[idx], self._init_agent(position_new))

    def optimization_step(self):
        # viruses diffusion
        self.__viruses_diffusion__()

        # host cells infection
        self.__host_cell_infection__()

        # immune response
        self.__immune_response__()
