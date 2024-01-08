import numpy as np

from ..helpers import (
    get_levy_flight_step,
    parse_obj_doc  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Jaya, LeviFlightJayaSwarmOptimizationConfig


class LeviFlightJayaSwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Levy-Flight Jaya Swarm Optimization algorithm.

    Args:
        config (LeviFlightJayaSwarmOptimizationConfig): an instance of LeviFlightJayaSwarmOptimizationConfig class.
            {parse_obj_doc(LeviFlightJayaSwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Iacca, G., dos Santos Junior, V.C. and de Melo, V.V., 2021. An improved Jaya optimization
        algorithm with Lévy flight. Expert Systems with Applications, 165, p.113902.
    """
    def __init__(self, config: LeviFlightJayaSwarmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __evolve__(self, idx: int, jaya: Jaya, g_worst_pos: list[float]) -> Jaya:
        position = np.array(jaya.position)

        L1 = get_levy_flight_step(multiplier=1.0, beta=1.8, case=-1)
        L2 = get_levy_flight_step(multiplier=1.0, beta=1.8, case=-1)
        pos_new = position + np.abs(L1) * (self._best_agent.position - np.abs(position)) - np.abs(L2) * (
            np.array(g_worst_pos) - np.abs(position)
        )
        agent = Jaya(**self._init_agent(pos_new).model_dump())
        return self._greedy_select_agent(jaya, agent)

    def optimization_step(self):
        g_worst_pos = np.array(self._worst_agent.position)
        self._population = self._solve_mode_process(self.__evolve__, g_worst_pos)
