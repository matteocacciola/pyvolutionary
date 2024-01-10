import numpy as np

from ..helpers import (
    get_levy_flight_step,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Pollinator, FlowerPollinationAlgorithmOptimizationConfig


class FlowerPollinationAlgorithmOptimization(OptimizationAbstract):
    """
    Implementation of the Flower Pollination Algorithm Optimization.

    Args:
        config (FlowerPollinationAlgorithmOptimizationConfig): an instance of
            FlowerPollinationAlgorithmOptimizationConfig class.
            {parse_obj_doc(FlowerPollinationAlgorithmOptimizationConfig)}

    Bibliography
    ----------
    [1] Yang, X.S., 2012, September. Flower pollination algorithm for global optimization. In International
        conference on unconventional computing and natural computation (pp. 240-249). Springer, Berlin, Heidelberg.
    """
    def __init__(self, config: FlowerPollinationAlgorithmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def evolve(pollinator: Pollinator, idx: int) -> Pollinator:
            position = np.array(pollinator.position)
            if np.random.uniform() < p_s:
                levy = get_levy_flight_step(multiplier=levy_multiplier, size=n_dims, case=-1)
                pos_new = position + 1.0 / np.sqrt(current_cycle) * levy * (position - best_position)
            else:
                id1, id2 = np.random.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)
                pos_new = position + np.random.uniform() * (
                    np.array(self._population[id1].position) - np.array(self._population[id2].position)
                )
            agent = Pollinator(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(agent, pollinator)

        n_dims = self._task.space_dimension
        pop_size = self._config.population_size
        current_cycle = self._current_cycle

        p_s = self._config.p_s
        levy_multiplier = self._config.levy_multiplier

        best_position = np.array(self._best_agent.position)

        self._population = [evolve(pollinator, idx) for idx, pollinator in enumerate(self._population)]
