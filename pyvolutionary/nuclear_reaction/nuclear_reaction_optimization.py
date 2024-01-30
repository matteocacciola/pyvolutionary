import math
from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import NuclearReactionOptimizationConfig, NuclearReaction


class NuclearReactionOptimization(OptimizationAbstract):
    """
    Implementation of the Nuclear Reaction Optimization algorithm.

    Args:
        config (NuclearReactionOptimizationConfig): an instance of NuclearReactionOptimizationConfig class.
            {parse_obj_doc(NuclearReactionOptimizationConfig)}

    Bibliography
    ----------
    [1] Wei, Z., Huang, C., Wang, X., Han, T. and Li, Y., 2019. Nuclear reaction optimization: A novel and powerful
        physics-based algorithm for global optimization. IEEE Access, 7, pp.66084-66109.
    [2] Wei, Z.L., Zhang, Z.R., Huang, C.Q., Han, B., Tang, S.Q. and Wang, L., 2019, June. An Approach Inspired from
        Nuclear Reaction Processes for Numerical Optimization. In Journal of Physics: Conference Series (Vol. 1213,
        No. 3, p. 032009). IOP Publishing.
    """
    def __init__(self, config: NuclearReactionOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = NuclearReactionOptimizationConfig(**parameters)

    def optimization_step(self):
        def nfi_phase(idx: int, reaction: NuclearReaction) -> NuclearReaction:
            position = np.array(reaction.position)
            i1 = np.random.choice(list(set(range(0, pop_size)) - {idx}), replace=False)
            pos_rand_agent = np.array(self._population[i1].position)
            Nei = (position + pos_rand_agent) / 2  # Calculate neutron vector Nei by Eq. (2)
            # Update based on Eq. 6 (np.random.uniform() <= Pb) or Eq. 3
            if np.random.uniform() <= Pfi:
                rnd = np.random.uniform()
                pos_rand_agent = position if rnd <= Pb else pos_rand_agent
                pos_new = g_best_position if rnd <= Pb else position
                offset = np.random.uniform() * g_best_position - round(np.random.random() + (
                    1 if np.random.uniform() <= Pb else 2
                )) * Nei
            # Update based on Eq. 9
            else:
                pos_new = position
                offset = 0
            xichma = (np.log(epoch) * 1.0 / epoch) * np.abs(np.subtract(pos_rand_agent, g_best_position))
            gauss = np.array([np.random.normal(pos_new[j], xichma[j]) for j in range(n_dims)])
            agent = NuclearReaction(**self._init_agent(gauss + offset).model_dump())
            return self._greedy_select_agent(reaction, agent)

        def nfu_phase(idx: int, reaction: NuclearReaction) -> NuclearReaction:
            position = np.array(reaction.position)
            i1, i2 = np.random.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)
            pos_i1 = np.array(self._population[i1].position)
            pos_i2 = np.array(self._population[i2].position)
            # Generate fusion nucleus
            if (ranked_pop[idx] * 1.0 / pop_size) < np.random.random():
                t1 = np.random.uniform() * (position - g_best_position)
                t2 = np.random.uniform() * (pos_i2 - g_best_position)
                temp2 = pos_i1 - pos_i2
                pos_new = position + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
            else:
                f = epochs - epoch if np.random.uniform() > 0.5 else epoch
                # Eq. 22 (if pos_i1 == pos_i2), or Eq. 16, 17
                pos_new = (
                    position + alpha * levy_b * (position - g_best_position)
                ) if np.all(pos_i1 == pos_i2) else (
                    position - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) * f / epochs + 1) * (pos_i1 - pos_i2)
                )
            agent = NuclearReaction(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(reaction, agent)
        
        def fusion_phase(idx: int, reaction: NuclearReaction) -> NuclearReaction:
            position = np.array(reaction.position)
            i1, i2 = np.random.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)
            pos_i1 = np.array(self._population[i1].position)
            pos_i2 = np.array(self._population[i2].position)
            if (ranked_pop[idx] * 1.0 / pop_size) < np.random.random():
                t1 = np.random.uniform() * (pos_i1 - g_best_position)
                t2 = np.random.uniform() * (pos_i2 - g_best_position)
                temp2 = pos_i1 - pos_i2
                pos_new = position + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
            else:
                f = epochs - epoch if np.random.uniform() > 0.5 else epoch
                # Eq. 22 (if pos_i1 == pos_i2), or Eq. 16, 17
                pos_new = (
                    position + alpha * levy_b * (position - g_best_position)
                ) if np.all(pos_i1 == pos_i2) else (
                    position - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) * f / epochs + 1) * (pos_i1 - pos_i2)
                )
            agent = NuclearReaction(**self._init_agent(pos_new).model_dump())
            return self._greedy_select_agent(reaction, agent)

        xichma_u = ((math.gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2)) / (
            math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2)
        )) ** (1.0 / 1.5)
        levy_b = (np.random.normal(0, xichma_u ** 2)) / (np.sqrt(np.abs(np.random.normal(0, 1))) ** (1.0 / 1.5))
        Pb, Pfi = np.random.uniform(size=2)
        freq = 0.05
        alpha = 0.01

        pop_size = self._config.population_size
        epoch = self._current_cycle
        epochs = self._config.max_cycles
        n_dims = self._task.space_dimension
        g_best_position = np.array(self._best_agent.position)

        # NFi phase
        self._population = [nfi_phase(idx, reaction) for idx, reaction in enumerate(self._population)]

        # NFu phase, i.e. ionization stage: calculate the population through Eq. (10)
        ranked_pop = np.argsort([self._population[i].cost for i in range(pop_size)])
        self._population = [nfu_phase(idx, reaction) for idx, reaction in enumerate(self._population)]

        # Fusion
        ranked_pop = np.argsort([self._population[i].cost for i in range(pop_size)])
        self._population = [fusion_phase(idx, reaction) for idx, reaction in enumerate(self._population)]
