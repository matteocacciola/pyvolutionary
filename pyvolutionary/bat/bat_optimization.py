import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Bat, BatOptimizationConfig


class BatOptimization(OptimizationAbstract):
    """
    Implementation of the Bat Optimization algorithm.

    Args:
        config (BatOptimizationConfig): an instance of BatOptimizationConfig class.
            {parse_obj_doc(BatsOptimizationConfig)}

    Bibliography
    ----------
    [1] Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for
        optimization" (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74.
    """

    def __init__(self, config: BatOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def _init_agent(
        self,
        position: list[float] | np.ndarray | None = None,
        velocity: list[float] | np.ndarray | None = None,
        loudness: float | None = None,
        pulse_rate: float | None = None,
    ) -> Bat:
        agent = super()._init_agent(position)
        velocity = self._correct_position(velocity if velocity is not None else self._uniform_position().tolist())

        return Bat(
            **agent.model_dump(),
            velocity=velocity,
            loudness=loudness if loudness is not None else np.random.uniform(*self._config.loudness),
            pulse_rate=pulse_rate if pulse_rate is not None else np.random.uniform(*self._config.pulse_rate),
        )

    def _greedy_select_agent(self, agent: Bat, new_agent: Bat) -> Bat:
        """
        Perform the greedy selection between the current agent and the new one. The greedy selection is performed by
        comparing the costs of each agent. The one with the lowest cost is kept.
        :param agent: the current agent
        :param new_agent: the new agent
        :return: the best agent
        :rtype: Bat
        """
        return new_agent if new_agent.cost < agent.cost and np.random.random() < agent.loudness else agent

    def optimization_step(self):
        mean_a = np.mean([bat.loudness for bat in self._population])

        pf_min, pf_max = self._config.pulse_frequency
        best_position = np.array(self._best_agent.position)
        for idx, bat in enumerate(self._population):
            velocity = bat.velocity + np.random.uniform(pf_min, pf_max) * (np.array(bat.position) - best_position)

            # Local Search around g_best position
            position = best_position + mean_a * np.random.normal(-1, 1) \
                if np.random.random() > bat.pulse_rate else bat.position + velocity
            self._population[idx] = self._greedy_select_agent(
                bat, self._init_agent(position, velocity, bat.loudness, bat.pulse_rate)
            )
