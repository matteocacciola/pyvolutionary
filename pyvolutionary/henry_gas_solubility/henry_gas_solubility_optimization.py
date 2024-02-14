from itertools import chain
from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Gas, HenryGasSolubilityOptimizationConfig


class HenryGasSolubilityOptimization(OptimizationAbstract):
    """
    Implementation of the Henry Gas Solubility Optimization algorithm.

    Args:
        config (HenryGasSolubilityOptimizationConfig): an instance of ElephantHerdOptimizationConfig class.
            {parse_obj_doc(ElephantHerdOptimizationConfig)}

    Bibliography
    ----------
    [1] Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. Henry gas solubility
        optimization: A novel physics-based algorithm. Future Generation Computer Systems, 101, pp.646-667.
    """
    def __init__(self, config: HenryGasSolubilityOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__T0 = 298.15
        self.__beta = 1.0
        self.__alpha = 1
        self.__epsilon = 0.05
        self.__groups: list[list[Gas]] = []
        self.__p_best: list[Gas] = []
        self.__n_elements: int = 0

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = HenryGasSolubilityOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__n_elements = int(self._config.population_size / self._config.n_clusters)

    def after_initialization(self):
        self.__groups = self._generate_group_population(self._config.n_clusters, self.__n_elements, False)
        self.__p_best = [best_agent(group) for group in self.__groups]

    def optimization_step(self):
        def evolve_group_element(p_best_i: Gas, element: Gas) -> Gas:
            pos = np.array(element.position)
            p_best_pos = np.array(p_best_i.position)
            F = -1.0 if np.random.uniform() < 0.5 else 1.0
            K_j = 5E-2 * np.random.uniform()
            P_ij = 100.0 * np.random.uniform()
            C_j = 1E-2 * np.random.uniform()
            # Update the solubility of each gas using Eq.9
            S_ij = K_j * np.exp(-C_j * (1.0 / np.exp(-cycle_ratio) - 1.0 / T0)) * P_ij
            gama = beta * np.exp(- ((p_best_i.cost + epsilon) / (element.cost + epsilon)))
            pos_new = pos + F * np.random.uniform() * gama * (p_best_pos - pos) + (
                F * np.random.uniform() * alpha * (S_ij * best_pos - pos)
            )
            return self._greedy_select_agent(element, Gas(**self._init_agent(pos_new).model_dump()))

        pop_size = len(self._population)
        cycle_ratio = self._current_cycle / self._config.max_cycles
        T0, alpha, beta, epsilon = self.__T0, self.__alpha, self.__beta, self.__epsilon
        p_best, best_pos = self.__p_best, np.array(self._best_agent.position)

        self.__groups = [
            [evolve_group_element(p_best, element) for element in group] for p_best, group in zip(p_best, self.__groups)
        ]
        self._population = list(chain.from_iterable(self.__groups))

        # Rank and select the number of worst agents using Eq. 11
        N_w = int(pop_size * (np.random.uniform(0, 0.1) + 0.1))
        pop_idx = np.argsort([x.cost for x in self._population])[pop_size - N_w:]

        # Update the position of the worst agents using Eq. 12
        self._population = [self._greedy_select_agent(
            element, Gas(**self._init_agent().model_dump())
        ) if idx in pop_idx else element for idx, element in enumerate(self._population)]
        self.after_initialization()
