from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Knowledge, GainingSharingKnowledgeOptimizationConfig


class GainingSharingKnowledgeOptimization(OptimizationAbstract):
    """
    Implementation of the Gaining Sharing Knowledge-based Optimization algorithm.

    Args:
        config (GainingSharingKnowledgeOptimizationConfig): an instance of GainingSharingKnowledgeOptimizationConfig
            class.
            {parse_obj_doc(GainingSharingKnowledgeOptimizationConfig)}

    Bibliography
    ----------
    [1] Mohamed, A.W., Hadi, A.A. and Mohamed, A.K., 2020. Gaining-sharing knowledge based algorithm for solving
        optimization problems: a novel nature-inspired algorithm. International Journal of Machine Learning and
        Cybernetics, 11(7), pp.1501-1529.
    """
    def __init__(self, config: GainingSharingKnowledgeOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__v: np.ndarray | None = None
        self.__pbest: list[Knowledge] = []

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = GainingSharingKnowledgeOptimizationConfig(**parameters)

    def optimization_step(self):
        def get_prev_next(idx: int) -> tuple[int, int]:
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                return idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            if idx == pop_size - 1:
                return idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            return idx - 1, idx + 1

        def junior_gaining_sharing_knowledge(idx: int, agent: Knowledge) -> np.ndarray:
            rand_idx = np.random.choice(list(set(range(0, pop_size)) - {prevs[idx], idx, nexts[idx]}))
            pos = np.array(agent.position)
            if np.random.uniform() > kr:
                return pos.copy()
            previ, nexti = prevs[idx], nexts[idx]
            previ_pos = np.array(self._population[previ].position)
            nexti_pos = np.array(self._population[nexti].position)
            rand_pos = np.array(self._population[rand_idx].position)
            return pos + kf * (previ_pos - nexti_pos + (rand_pos - pos) * (
                1 if self._population[rand_idx].cost < agent.cost else -1
            ))

        def senior_gaining_sharing_knowledge(idx: int, agent: Knowledge) -> np.ndarray:
            pos = np.array(agent.position)
            if np.random.uniform() > kr:
                return pos.copy()
            rand_best = np.random.choice(list(set(range(0, id1)) - {idx}))
            rand_worst = np.random.choice(list(set(range(id2, pop_size)) - {idx}))
            rand_mid = np.random.choice(list(set(range(id1, id2)) - {idx}))
            pos_rand_best = np.array(self._population[rand_best].position)
            pos_rand_worst = np.array(self._population[rand_worst].position)
            pos_rand_mid = np.array(self._population[rand_mid].position)
            return pos + kf * (pos_rand_best - pos_rand_worst + (pos_rand_mid - pos) * (
                1 if self._population[rand_mid].cost < agent.cost else -1
            ))

        def evolve(idx: int, agent: Knowledge) -> Knowledge:
            pos = np.where(
                list(range(n_dims)) < np.repeat(dd, n_dims),
                junior_gaining_sharing_knowledge(idx, agent),
                senior_gaining_sharing_knowledge(idx, agent)
            )
            return self._greedy_select_agent(agent, Knowledge(**self._init_agent(pos).model_dump()))

        p, kf, kr, kg = self._config.p, self._config.kf, self._config.kr, self._config.kg
        pop_size = self._config.population_size
        cycle_ratio = self._current_cycle / self._config.max_cycles

        prevs, nexts = zip(*[get_prev_next(i) for i in range(pop_size)])
        n_dims = self._task.space_dimension
        dd = int(n_dims * (1 - cycle_ratio) ** kg)

        id1 = int(p * pop_size)
        id2 = int(pop_size * (1 - p))

        self._population = [evolve(idx, agent) for idx, agent in enumerate(self._population)]