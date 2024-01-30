from typing import Any
import numpy as np

from ..helpers import (
    best_agent,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Egret, EgretSwarmOptimizationConfig


class EgretSwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Egret Swarm Optimization algorithm.

    Args:
        config (EgretSwarmOptimizationConfig): an instance of EgretSwarmOptimizationConfig class.
            {parse_obj_doc(EgretSwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Chen, Z., Francis, A., Li, S., Liao, B., Xiao, D., Ha, T. T., ... & Cao, X. (2022). Egret Swarm Optimization
        Algorithm: An Evolutionary Computation Approach for Model Free Optimization. Biomimetics, 7(4), 144.
    """
    def __init__(self, config: EgretSwarmOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__ebest: list[Egret] = []

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = EgretSwarmOptimizationConfig(**parameters)

    def _init_agent(
        self,
        position: list[Any] | np.ndarray | None = None,
        m: list[float] | None = None,
        v: list[float] | None = None,
        weights: list[float] | None = None,
    ) -> Egret:
        agent = super()._init_agent(position=position)
        n_dims = self._task.space_dimension
        weights = weights if weights is not None else np.random.uniform(-1., 1., n_dims).tolist()
        return Egret(
            **agent.model_dump(),
            weights=weights,
            m=m if m is not None else np.zeros(n_dims).tolist(),
            v=v if v is not None else np.zeros(n_dims).tolist(),
        )

    def after_initialization(self):
        self.__ebest = self._population.copy()

    def optimization_step(self):
        def evolve(egret: Egret, ebest: Egret) -> tuple[Egret, Egret]:
            local_pos = np.array(ebest.position)
            pos = np.array(egret.position)
            # Individual Direction
            p_d = (local_pos - pos) * (ebest.cost - egret.cost)
            d_p = p_d / ((np.sum(p_d) + self.EPS) ** 2) + egret.g
            # Group Direction
            c_d = (best_position - pos) * (best_cost - egret.cost)
            d_g = c_d / ((np.sum(c_d) + self.EPS) ** 2) + best_g
            # Gradient Estimation
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            g = (1 - r1 - r2) * egret.g + r1 * d_p + r2 * d_g
            g /= (np.sum(g) + self.EPS)
            m = (beta1 * np.array(egret.m) + (1 - beta1) * g).tolist()
            v = (beta2 * np.array(egret.v) + (1 - beta2) * g ** 2).tolist()
            weights = np.array(egret.weights) - np.array(m) / (np.sqrt(v) + self.EPS)
            # Advice Forward
            x_0 = self._init_agent(
                pos + np.exp(-1.0 / (0.1 * epochs)) * 0.1 * hop * g,
                m=m,
                v=v,
                weights=weights,
            )
            # Random Search
            x_n = self._init_agent(
                pos + np.tan(np.random.uniform(-np.pi / 2, np.pi / 2, n_dims)) * hop / epoch * 0.5,
                m=m,
                v=v,
                weights=weights,
            )
            # Encircling Mechanism
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            x_m = self._init_agent(
                (1 - r1 - r2) * pos + r1 * (local_pos - pos) + r2 * (best_position - pos),
                m=m,
                v=v,
                weights=weights,
            )
            # Discriminant Condition
            x_best = best_agent([x_0, x_n, x_m])
            return self._greedy_select_agent(egret, x_best), self._greedy_select_agent(ebest, x_best)

        beta1 = 0.9
        beta2 = 0.99
        hop = self._task.bandwidth()

        best_position = np.array(self._best_agent.position)
        best_cost = self._best_agent.cost
        best_g = self._best_agent.g

        epoch = self._current_cycle
        epochs = self._config.max_cycles
        n_dims = self._task.space_dimension

        # merge the population and the archive, sort them by cost and update the population with the best n_ants
        self._population, self.__ebest = map(
            lambda x: list(x),
            zip(*[evolve(particle, ebest) for particle, ebest in zip(self._population, self.__ebest)]),
        )
