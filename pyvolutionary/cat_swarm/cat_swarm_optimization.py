import numpy as np

from ..helpers import (
    best_agent,
    parse_obj_doc,  # type: ignore
    roulette_wheel_index,
)
from ..abstract import OptimizationAbstract
from .models import Cat, CatSwarmOptimizationConfig


class CatSwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Cat Swarm Optimization algorithm.

    Args:
        config (CatSwarmOptimizationConfig): an instance of CatSwarmOptimization class.
            {parse_obj_doc(CatSwarmOptimization)}

    Bibliography
    ----------
    [1] Chu, S.C., Tsai, P.W. and Pan, J.S., 2006, August. Cat swarm optimization. In Pacific Rim international
        conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
    """
    def __init__(self, config: CatSwarmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def _init_agent(
        self,
        position: list[float] | np.ndarray | None = None,
        velocity: list[float] | np.ndarray | None = None,
        flag: bool | None = None,
    ) -> Cat:
        agent = super()._init_agent(position=position)

        velocity = self._uniform_position() if velocity is None else (
            velocity.tolist() if isinstance(velocity, np.ndarray) else velocity
        )
        flag = np.random.uniform() < self._config.mixture_ratio if flag is None else flag

        return Cat(**agent.model_dump(), velocity=velocity, flag=flag)

    def optimization_step(self):
        def seeking_clone(c: Cat) -> Cat:
            pos = np.array(c.position)
            jdx = np.random.choice(range(0, n_dims), int(cdc * n_dims), replace=False)
            pos_new = np.where(np.random.random(n_dims) < 0.5, pos * (1 + srd), pos * (1 - srd))
            pos_new[jdx] = pos[jdx]
            return self._init_agent(pos_new, c.velocity, c.flag)

        def seeking_mode(cat: Cat) -> list[float]:
            cloned = [cat.model_copy() for _ in range(smp - 1)] if spc else self._generate_agents(self._config.smp)
            candidates = [cat.model_copy()] if self._config.spc else []
            candidates += [seeking_clone(Cat(**cat.model_dump())) for cat in cloned]
            if selected_strategy == 0:  # best fitness-self
                return best_agent(candidates).position
            if selected_strategy == 1:  # tournament
                k_way = 4
                idx = np.random.choice(range(0, self._config.smp), k_way, replace=False)
                cats_k_way = [candidates[_] for _ in idx]
                return best_agent(cats_k_way).position
            if selected_strategy == 2:  # roulette wheel selection
                idx = roulette_wheel_index(np.array([c.cost for c in candidates]))
                return candidates[idx].position
            idx = np.random.choice(range(0, len(candidates)))  # random
            return candidates[idx].position

        def evolve(cat: Cat) -> Cat:
            pos = np.array(cat.position)
            pos_new = (
                pos + w * np.array(cat.velocity) + np.random.uniform() * self._config.c1 * (best_position - pos)
                if cat.flag else seeking_mode(cat)
            )
            return self._init_agent(pos_new, cat.velocity)

        n_dims = self._task.space_dimension
        smp = self._config.smp
        spc = self._config.spc
        srd = self._config.srd
        cdc = self._config.cdc
        selected_strategy = self._config.selected_strategy
        w_min, w_max = self._config.w
        w = (self._config.max_cycles - self._current_cycle) / self._config.max_cycles * (w_max - w_min) + w_min
        best_position = np.array(self._best_agent.position)

        self._population = [evolve(cat) for cat in self._population]
