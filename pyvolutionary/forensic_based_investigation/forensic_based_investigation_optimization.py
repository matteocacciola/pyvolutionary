from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import ForensicBasedInvestigationOptimizationConfig, Detective


class ForensicBasedInvestigationOptimization(OptimizationAbstract):
    """
    Implementation of the Forensic Based Investigation Optimization algorithm.

    Args:
        config (ForensicBasedInvestigationOptimizationConfig): an instance of
            ForensicBasedInvestigationOptimizationConfig class.
            {parse_obj_doc(ForensicBasedInvestigationOptimizationConfig)}

    Bibliography
    ----------
    [1] Chou, J.S. and Nguyen, N.M., 2020. FBI inspired meta-optimization. Applied Soft Computing, 93, p.106339.
    """
    def __init__(self, config: ForensicBasedInvestigationOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = ForensicBasedInvestigationOptimizationConfig(**parameters)

    def optimization_step(self):
        def investigation_a1(idx: int, detective: Detective) -> Detective:
            pos = detective.position
            n_change = np.random.randint(0, n_dims)
            pos_nb1, pos_nb2 = map(
                lambda x: self._population[x].position,
                np.random.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False),
            )
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = pos.copy()
            pos_a[n_change] = pos[n_change] + (np.random.normal() - 0.5) * (
                pos[n_change] - (pos_nb1[n_change] + pos_nb2[n_change]) / 2
            )
            return self._greedy_select_agent(detective, Detective(**self._init_agent(pos_a).model_dump()))

        def investigation_a2(idx: int, detective: Detective) -> Detective:
            pos = np.array(detective.position)
            if np.random.uniform() <= prob[idx]:
                return detective
            pos_r1, pos_r2, pos_r3 = map(
                lambda x: np.array(self._population[x].position),
                np.random.choice(list(set(range(0, pop_size)) - {idx}), 3, replace=False)
            )
            pos_new = np.where(
                np.random.random(n_dims) < 0.5,
                best_pos + pos_r1 + np.random.uniform() * (pos_r2 - pos_r3),
                pos
            )
            return self._greedy_select_agent(detective, Detective(**self._init_agent(pos_new).model_dump()))

        def pursuing_b1(detective: Detective) -> Detective:
            pos = np.array(detective.position)
            pos_new = np.random.uniform() * pos + np.random.uniform() * (best_pos - pos)
            return self._greedy_select_agent(detective, Detective(**self._init_agent(pos_new).model_dump()))
        
        def pursuing_b2(idx: int, detective: Detective) -> Detective:
            pos = np.array(detective.position)
            rr = np.random.choice(list(set(range(0, pop_size)) - {idx}))
            pos_rr = np.array(self._population[rr].position)
            cost_rr = self._population[rr].cost
            r = np.random.uniform(0, 1, n_dims)
            r1 = np.random.uniform(0, 1, n_dims)
            # Eqs. (7), if detective.cost < self._population[rr].cost, and (8) in FBI Inspired Meta-Optimization
            if detective.cost > cost_rr:
                pos_b = pos_rr + r * (pos_rr - pos) + r1 * (best_pos - pos_rr)
            else:
                pos_b = pos + r * (pos - pos_rr) + r1 * (best_pos - pos)
            return self._greedy_select_agent(detective, Detective(**self._init_agent(pos_b).model_dump()))

        n_dims = self._task.space_dimension
        pop_size = self._config.population_size

        # investigation team - team A
        # Step A1
        best_pos = np.array(self._best_agent.position)
        self._population = [investigation_a1(idx, detective) for idx, detective in enumerate(self._population)]

        # Step A2
        list_cost = np.array([agent.cost for agent in self._population])
        min_c, max_c = np.min(list_cost), np.max(list_cost)
        prob = (max_c - list_cost) / (max_c - min_c + self.EPS)
        self._population = [investigation_a2(idx, detective) for idx, detective in enumerate(self._population)]

        # pursuing team - team B
        # step B1
        self._population = [pursuing_b1(detective) for detective in self._population]

        # step B2
        self._population = [pursuing_b2(idx, detective) for idx, detective in enumerate(self._population)]
