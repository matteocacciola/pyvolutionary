import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import ZebraOptimizationConfig, Zebra


class ZebraOptimization(OptimizationAbstract):
    """
    Implementation of the Zebra Optimization algorithm.

    Args:
        config (ZebraOptimizationConfig): an instance of ZebraOptimizationConfig class.
            {parse_obj_doc(ZebraOptimizationConfig)}

    Bibliography
    ----------
    [1] Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Zebra optimization algorithm: A new bio-inspired
        optimization algorithm for solving optimization algorithm. IEEE Access, 10, 49445-49473.
    """
    def __init__(self, config: ZebraOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def foraging(zebra: Zebra) -> Zebra:
            pos = np.array(zebra.position)
            r1 = np.round(1 + np.random.random())
            pos_new = pos + np.random.random(n_dims) * (best_pos - r1 * pos)  # Eq. 3
            agent = Zebra(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(agent, zebra)

        def defense_strategy(zebra: Zebra) -> Zebra:
            pos = np.array(zebra.position)
            # strategy 1: the lion attacks the zebra and thus the zebra chooses an escape strategy OR
            # strategy 2: other predators attack the zebra and the zebra will choose the offensive strategy
            pos_new = (
                pos + 0.1 * (2 + np.random.random(n_dims) - 1) * (1 - cycle_ratio) * pos
            ) if np.random.random() < 0.5 else pos + np.random.random(n_dims) * (
                pos_kk - np.random.randint(1, 3) * pos
            )
            agent = Zebra(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(agent, zebra)

        n_dims = self._task.space_dimension
        cycle_ratio = self._current_cycle / self._config.max_cycles

        # phase 1: foraging behaviour
        best_pos = np.array(self._best_agent.position)
        self._population = [foraging(zebra) for zebra in self._population]

        # phase 2: defense strategies against predators
        kk = np.random.permutation(self._config.population_size)[0]
        pos_kk = np.array(self._population[kk].position)
        self._population = [defense_strategy(zebra) for zebra in self._population]
