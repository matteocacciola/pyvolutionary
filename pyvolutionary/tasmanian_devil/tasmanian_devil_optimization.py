import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import TasmanianDevil, TasmanianDevilOptimizationConfig


class TasmanianDevilOptimization(OptimizationAbstract):
    """
    Implementation of the Tasmanian Devil Optimization algorithm.

    Args:
        config (TasmanianDevilOptimizationConfig): an instance of TasmanianDevilOptimizationConfig class.
            {parse_obj_doc(TasmaniaDevilOptimizationConfig)}

    Bibliography
    ----------
    [1] Dehghani, M., Hubálovský, Š., & Trojovský, P. (2022). Tasmanian devil optimization: a new bio-inspired
        optimization algorithm for solving optimization algorithm. IEEE Access, 10, 19599-19620.
    """
    def __init__(self, config: TasmanianDevilOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def __strategy__(self, idx: int, tasmanian_devil: TasmanianDevil) -> TasmanianDevil:
        """
        Application of strategy.
        :param idx: index of the agent
        :param tasmanian_devil: current agent
        :return: picked agent
        :rtype: TasmanianDevil
        """
        pop_size = self._config.population_size
        n_dims = self._task.space_dimension

        pos = np.array(tasmanian_devil.position)

        kk = np.random.choice(list(set(range(0, pop_size)) - {idx}))

        pos_new = pos + np.random.random(n_dims) * (
            pos - np.random.randint(1, 3) * pos
        ) if self._population[kk].cost < tasmanian_devil.cost else pos + np.random.random(n_dims) * (
            pos - np.array(self._population[kk].position)
        )
        agent = self._init_agent(self._correct_position(pos_new))
        return self._greedy_select_agent(tasmanian_devil, agent)

    def optimization_step(self):
        epoch = self._cycles
        epochs = self._config.max_cycles
        n_dims = self._task.space_dimension

        for idx, tasmanian_devil in enumerate(self._population):
            # phase 1: hunting feeding
            if np.random.random() > 0.5:
                # strategy 1: feeding by eating carrion (exploration phase)
                # carrion selection using (3)
                self._population[idx] = self.__strategy__(idx, tasmanian_devil)
            else:
                # strategy 2: feeding by eating prey (exploitation phase)
                # stage 1: prey selection and attack it
                agent = self.__strategy__(idx, tasmanian_devil)
                pos = np.array(agent.position)

                # phase 2: prey chasing
                rr = 0.01 * (1 - epoch / epochs)  # Calculating the neighborhood radius using(9)
                pos_new = pos + (-rr + 2 * rr * np.random.random(n_dims)) * pos
                agent = self._init_agent(self._correct_position(pos_new))

                self._population[idx] = self._greedy_select_agent(tasmanian_devil, agent)
