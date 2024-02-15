from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Molecule, FicksLawOptimizationConfig


class FicksLawOptimization(OptimizationAbstract):
    """
    Implementation of the Fick's Law Optimization algorithm.

    Args:
        config (FicksLawOptimizationConfig): an instance of FicksLawOptimizationConfig class.
            {parse_obj_doc(FicksLawOptimizationConfig)}

    Bibliography
    ----------
    [1] Hashim, F. A., Mostafa, R. R., Hussien, A. G., Mirjalili, S., & Sallam, K. M. (2023). Fick’s Law Algorithm: A
        physical law-based algorithm for numerical optimization. Knowledge-Based Systems, 260, 110146.

    Side Notes
    ----------
    Despite the complexity of the algorithm, the performances are not so good, since the agents can potentially be stuck
    in local minima. The algorithm is not recommended for complex optimization problems.
    """
    def __init__(self, config: FicksLawOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__n1: int = 0
        self.__n2: int = 0
        self.__pop1: list[Molecule] = []
        self.__pop2: list[Molecule] = []
        self.__best1: Molecule | None = None
        self.__best2: Molecule | None = None
        self.__fsss: float = 0.0

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = FicksLawOptimizationConfig(**parameters)

    def after_initialization(self):
        self.__n1 = int(np.round(self._config.population_size / 2))
        self.__n2 = self._config.population_size - self.__n1
        self.__pop1 = self._population[:self.__n1].copy()
        self.__pop2 = self._population[self.__n1:].copy()
        self.__best1 = best_agent(self.__pop1)
        self.__best2 = best_agent(self.__pop2)
        self.__fsss = min(self.__best1.cost, self.__best2.cost)

    def optimization_step(self):
        def non_equilibrium_operator_1(molecule: Molecule, xm_factor: int, best_pos: np.ndarray) -> Molecule:
            pos = np.array(molecule.position)
            dfg = np.random.randint(1, 3)
            jj = -DD * xm_factor * (xm2 - xm1) / np.linalg.norm(best_pos - pos + self.EPS)
            return Molecule(**self._init_agent(
                best_pos + dfg * dof * np.random.random(n_dims) * (jj * best_pos - pos)
            ).model_dump())

        def non_equilibrium_operator_2(molecule: Molecule, best_pos: np.ndarray) -> Molecule:
            pos = np.array(molecule.position)
            tt = pos + dof * (np.random.random(n_dims) * bandwidth + lb)
            pp = np.random.random(n_dims)
            return Molecule(**self._init_agent(np.where(pp < 0.8, best_pos, np.where(pp >= 0.9, pos, tt))).model_dump())

        def equilibrium_operator(
            molecule: Molecule, best_pos: np.ndarray, best_cost: float, xm_: np.ndarray
        ) -> Molecule:
            pos = np.array(molecule.position)
            dfg = np.random.randint(1, 3)
            tttt = np.linalg.norm(best_pos - pos)
            jj = 0 if tttt == 0 else -DD * (best_pos - xm_) / tttt
            drf = np.exp(-jj / tf)
            ms = np.exp(-best_cost / molecule.cost + self.EPS)
            qeo = dfg * drf * np.random.random(n_dims)
            return Molecule(**self._init_agent(best_pos + qeo * pos + qeo * (ms * best_pos - pos)).model_dump())

        def steady_state_operator(molecule: Molecule, best_pos: np.ndarray, xm_: np.ndarray) -> Molecule:
            pos = np.array(molecule.position)
            dfg = np.random.randint(1, 3)
            tttt = np.linalg.norm(best_pos - pos)
            jj = 0 if tttt == 0 else -DD * (xm - xm_) / tttt
            drf = np.exp(-jj / tf)
            ms = np.exp(-fsss / molecule.cost + self.EPS)
            qg = dfg * drf * np.random.random(n_dims)
            return Molecule(**self._init_agent(g_best + qg * pos + qg * (ms * best_pos - pos)).model_dump())

        C1, C2, C3, C4, C5 = self._config.C1, self._config.C2, self._config.C3, self._config.C4, self._config.C5
        DD = self._config.DD
        
        n1, n2, fsss = self.__n1, self.__n2, self.__fsss
        g_best = np.array(self._best_agent.position)
        pop1, pop2, best1, best2 = self.__pop1, self.__pop2, self.__best1, self.__best2
        best1_pos, best2_pos = np.array(best1.position), np.array(best2.position)

        epoch, epochs = self._current_cycle, self._config.max_cycles
        n_dims = self._task.space_dimension
        lb, ub = self._task.get_bounds()
        bandwidth = self._task.bandwidth()

        xm1 = np.mean(np.array([agent.position for agent in self.__pop1]), axis=0)
        xm2 = np.mean(np.array([agent.position for agent in self.__pop2]), axis=0)
        xm = np.mean(np.array([agent.position for agent in self._population]), axis=0)
        tf = np.sinh((epoch + 1) / epochs) ** C1
        if tf < 0.9:
            dof = np.exp(-(C2 * tf - np.random.random())) ** C2
            tdo = C5 * tf - np.random.random()
            if tdo < np.random.random():
                m1n, m2n = C3 * n1, C4 * n1
                nt12 = int(np.round((m2n - m1n) * np.random.random() + m1n))
                pop_new = [non_equilibrium_operator_1(molecule, 1, best2_pos) for molecule in pop1[:nt12]] + (
                    [non_equilibrium_operator_2(molecule, best1_pos) for molecule in pop1[nt12:n1]]
                ) + [self._init_agent(best2_pos + dof * (np.random.random(n_dims) * bandwidth + lb)) for _ in range(n2)]
            else:
                m1n, m2n = 0.1 * n2, 0.2 * n2
                nt12 = int(np.round((m2n - m1n) * np.random.random() + m1n))
                pop_new = [non_equilibrium_operator_1(molecule, -1, best1_pos) for molecule in pop2[:nt12]] + (
                    [non_equilibrium_operator_2(molecule, best2_pos) for molecule in pop2[nt12:n2]]
                ) + [self._init_agent(best1_pos + dof * (np.random.random(n_dims) * bandwidth + lb)) for _ in range(n1)]
        elif tf <= 1:  # Equilibrium operator (EO)
            pop_new = [equilibrium_operator(molecule, best1_pos, best1.cost, xm1) for molecule in pop1[:n1]] + (
                [equilibrium_operator(molecule, best2_pos, best2.cost, xm2) for molecule in pop2[:n2]]
            )
        else:  # Steady state operator (SSO)
            pop_new = [steady_state_operator(molecule, best1_pos, xm1) for molecule in pop1[:n1]] + (
                [steady_state_operator(molecule, g_best, xm2) for molecule in pop2[:n2]]
            )

        self._greedy_select_population(pop_new)
        self.after_initialization()
