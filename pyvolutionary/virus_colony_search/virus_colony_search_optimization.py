import numpy as np

from ..helpers import (
    sort_by_cost,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import VirusColonySearchOptimizationConfig, Virus


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

    def optimization_step(self):
        def calculate_x_mean() -> list[float]:
            pop = self._population.copy()
            sort_by_cost(pop)
            positions = [np.array(agent.position) for agent in pop[:n_best]]
            weight = np.log1p(n_best + 1) / (n_best * np.log1p(n_best + 1) - np.log1p(np.prod(range(1, n_best + 1))))
            return weight * np.sum(positions, axis=0) / n_best

        def virus_diffusion(virus: Virus) -> Virus:
            position = np.array(virus.position)
            sigma_diffusion = (np.log1p(cycle) / max_cycles) * (position - best_position)
            gauss = np.random.normal(np.random.normal(best_position, np.abs(sigma_diffusion)))
            position_new = gauss + np.random.uniform() * best_position - np.random.uniform() * position
            agent = Virus(**self._init_agent(position_new).model_dump())
            return self._greedy_select_agent(virus, agent)

        def host_cell_infection(virus: Virus) -> Virus:
            agent = Virus(**self._init_agent(
                np.array(calculate_x_mean()) + sigma_infection * np.random.normal(0, 1, n_dims)
            ).model_dump())
            return self._greedy_select_agent(virus, agent)

        def immune_response(idx: int, virus: Virus) -> Virus:
            position = np.array(virus.position)
            pr = (n_dims - idx + 1) / n_dims
            id1, id2 = np.random.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)
            temp = np.array(self._population[id1].position) - np.random.uniform() * (
                np.array(self._population[id2].position) - position
            )
            position_new = np.where(np.random.random(n_dims) < pr, position, temp)
            agent = Virus(**self._init_agent(position_new).model_dump())
            return self._greedy_select_agent(self._population[idx], agent)

        max_cycles = self._config.max_cycles
        cycle = self._current_cycle
        best_position = np.array(self._best_agent.position)

        n_best = self.__n_best
        n_dims = self._task.space_dimension
        pop_size = self._config.population_size

        # viruses diffusion
        self._population = [virus_diffusion(virus) for virus in self._population]

        # host cells infection
        sigma_infection = self._config.sigma * (1 - self._current_cycle / self._config.max_cycles)
        self._population = [host_cell_infection(virus) for virus in self._population]

        # immune response
        self._population = [immune_response(idx, virus) for idx, virus in enumerate(self._population)]
