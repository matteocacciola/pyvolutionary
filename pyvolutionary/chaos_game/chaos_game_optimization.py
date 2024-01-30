from itertools import chain
from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import ChaosGameOptimizationConfig, Seed


class ChaosGameOptimization(OptimizationAbstract):
    """
    Implementation of the Chaos Game algorithm.

    Args:
        config (ChaosGameOptimizationConfig): an instance of ChaosGameOptimizationConfig class.
            {parse_obj_doc(ChaosGameOptimizationConfig)}

    Bibliography
    ----------
    [1] Talatahari, S. and Azizi, M., 2021. Chaos Game Optimization: a novel metaheuristic algorithm. Artificial
        Intelligence Review, 54(2), pp.917-1004.
    """
    def __init__(self, config: ChaosGameOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = ChaosGameOptimizationConfig(**parameters)

    def optimization_step(self):
        def evolve(agent: Seed) -> list[Seed]:
            pos = np.array(agent.position)
            I = np.random.randint(1, 3, size=6)  # Beta and Gamma
            Ir = np.random.randint(0, 2, 5)
            # random groups
            rand_group_number = np.random.permutation(self._config.population_size)[0]
            rand_group = np.random.permutation(self._config.population_size)[:rand_group_number+1]
            groups_positions = [self._population[i].position for i in rand_group]
            # mean of random group
            mean_group_pos = np.mean(np.array(groups_positions), axis=0) if len(rand_group) > 1 else (
                np.array(groups_positions[0])
            )
            # Generate random values for Alfa
            alpha = np.vstack([
                np.random.rand(n_dims),
                2 * np.random.rand(n_dims) - 1,
                Ir[0] * np.random.rand(n_dims) + 1,
                Ir[1] * np.random.rand(n_dims) + (~Ir[1])
            ])
            # Select three rows randomly
            selected_alpha = alpha[np.random.randint(0, 4, size=3), :]
            new_seed_1 = Seed(**self._init_agent(
                pos + selected_alpha[0] * (I[0] * g_best_pos - I[1] * mean_group_pos)
            ).model_dump())
            new_seed_2 = Seed(**self._init_agent(
                g_best_pos + selected_alpha[1] * (I[2] * mean_group_pos - I[3] * pos)
            ).model_dump())
            new_seed_3 = Seed(**self._init_agent(
                mean_group_pos + selected_alpha[2] * (I[4] * g_best_pos - I[5] * pos)
            ).model_dump())
            new_seed_4 = Seed(**self._init_agent().model_dump())
            return [new_seed_1, new_seed_2, new_seed_3, new_seed_4]

        n_dims = self._task.space_dimension
        g_best_pos = np.array(self._best_agent.position)

        pop_new = list(chain.from_iterable(map(evolve, self._population)))

        # update population
        self._extend_and_trim_population(pop_new)
