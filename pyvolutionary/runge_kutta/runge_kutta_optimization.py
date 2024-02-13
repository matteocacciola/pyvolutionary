from typing import Any
import numpy as np

from ..models import Agent
from ..helpers import (
    best_agent_index,
    runge_kutta,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import RungeKuttaOptimizationConfig


class RungeKuttaOptimization(OptimizationAbstract):
    """
    Implementation of the Runge Kutta Optimization algorithm.

    Args:
        config (RungeKuttaOptimizationConfig): an instance of RungeKuttaOptimizationConfig class.
            {parse_obj_doc(RungeKuttaOptimizationConfig)}

    Bibliography
    ----------
    [1] Ahmadianfar, I., Heidari, A. A., Gandomi, A. H., Chu, X., & Chen, H. (2021). RUN beyond the metaphor: An
        efficient optimization algorithm based on Runge Kutta method. Expert Systems with Applications, 181, 115079.
    """
    def __init__(self, config: RungeKuttaOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = RungeKuttaOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve_runge_kutta(
            idx: int, agent: Agent
        ) -> tuple[Agent, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            current_pos = np.array(agent.position)

            # determine Delta X (Eqs. 11.1 to 11.3)
            gama = np.random.random() * (
                current_pos - np.random.uniform(0, 1, n_dims) * bandwidth
            ) * np.exp(-4 * epoch_ratio)
            stp = np.random.uniform(0, 1, n_dims) * ((best_pos - np.random.random() * x_average) + gama)
            delta_x = 2 * np.random.uniform(0, 1, n_dims) * np.abs(stp)

            # determine Three Random Indices of Solutions
            a, b, c = np.random.choice(list(set(range(0, pop_size)) - {idx}), 3, replace=False)
            pos_a = np.array(self._population[a].position)
            pos_b = np.array(self._population[b].position)
            pos_c = np.array(self._population[c].position)
            id_min_x = best_agent_index([self._population[a], self._population[b], self._population[c]])

            # determine Xb and Xw for using in Runge Kutta method
            xb, xw = np.array(self._population[id_min_x].position), current_pos
            if agent.cost < self._population[id_min_x].cost:
                xb, xw = xw, xb

            # Search Mechanism (SM) of RUN based on Runge Kutta Method
            SM = runge_kutta(xb, xw, delta_x)
            L = np.random.choice(range(0, 2), n_dims)
            xc = L * current_pos + (1 - L) * pos_a  # Eq. 17.3
            xm = L * best_pos + (1 - L) * best_pos  # Eq. 17.4
            r = np.random.choice([1, -1], n_dims)  # An integer number
            g = 2 * np.random.random()
            mu = 0.5 + 1 * np.random.uniform(0, 1, n_dims)

            # determine New Solution Based on Runge Kutta Method (Eq.18)
            pos_new = xm + r * SF[idx] * g * xm + SF[idx] * SM + mu * (pos_a - pos_b)
            if np.random.random() < 0.5:
                pos_new = xc + r * SF[idx] * g * xc + SF[idx] * SM + mu * (xm - xc)
            return self._greedy_select_agent(agent, self._init_agent(pos_new)), b, pos_a, pos_b, pos_c, delta_x

        def enhanced_solution_quality(
            idx: int,
            agent: Agent,
            b: np.ndarray,
            pos_a: np.ndarray,
            pos_b: np.ndarray,
            pos_c: np.ndarray,
            delta_x: np.ndarray
        ) -> Agent:
            w = b * (2 * np.random.uniform(0, 1, n_dims) - 1) * np.exp(
                -5 * np.random.random() * epoch_ratio
            )  # Eq.19-1
            r = np.floor(b * (2 * np.random.uniform(0, 1, 1) - 1))
            u = 2 * np.random.random(n_dims)
            # a, b, c = np.random.choice(list(set(range(0, pop_size)) - {idx}), 3, replace=False)
            x_ave = (pos_a + pos_b + pos_c) / 3  # Eq.19-2
            beta = np.random.random(n_dims)
            x_new1 = beta * best_pos + (1 - beta) * x_ave  # Eq.19-3
            x_new2 = np.where(
                w < 1,
                x_new1 + r * w * np.abs(np.random.normal(0, 1, n_dims) + (x_new1 - x_ave)),
                x_new1 - x_ave + r * w * np.abs(np.random.normal(0, 1, n_dims) + u * x_new1 - x_ave)
            )
            new_agent = self._init_agent(x_new2)
            if new_agent.cost < agent.cost:
                return new_agent
            if w[np.random.randint(0, n_dims)] > np.random.random():
                pos_new2 = np.array(self._task.initial_solution(x_new2))
                SM = runge_kutta(np.array(agent.position), pos_new2, delta_x)
                x_new3 = pos_new2 - np.random.random() * pos_new2 + SF[idx] * (SM + (2 * np.random.random(n_dims) * best_pos - pos_new2))  # Eq. 20
                return self._greedy_select_agent(agent, self._init_agent(x_new3))
            return agent

        def evolve(idx: int, agent: Agent) -> Agent:
            candidate, b, pos_a, pos_b, pos_c, delta_x = evolve_runge_kutta(idx, agent)
            if np.random.random() < 0.5:
                return enhanced_solution_quality(idx, candidate, b, pos_a, pos_b, pos_c, delta_x)
            return candidate

        pop_size = self._config.population_size
        n_dims = self._task.space_dimension

        best_agent = self._best_agent
        best_pos = np.array(best_agent.position)
        bandwidth = self._task.bandwidth()
        epoch_ratio = self._current_cycle / self._config.max_cycles

        f = 20 * np.exp(-12. * epoch_ratio)  # Eq.17.6
        SF = 2. * (0.5 - np.random.random(pop_size)) * f  # Eq.17.5
        x_average = np.mean(np.array([agent.position for agent in self._population]), axis=0)

        self._population = [evolve(idx, agent) for idx, agent in enumerate(self._population)]
