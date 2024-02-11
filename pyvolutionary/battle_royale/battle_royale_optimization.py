from typing import Any
import numpy as np
from scipy.spatial.distance import cdist

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import BattleRoyaleOptimizationConfig, Soldier


class BattleRoyaleOptimization(OptimizationAbstract):
    """
    Implementation of the Battle Royale Optimization algorithm.

    Args:
        config (BattleRoyaleOptimizationConfig): an instance of BattleRoyaleOptimizationConfig class.
            {parse_obj_doc(BattleRoyaleOptimizationConfig)}

    Bibliography
    ----------
    [1] Rahkar Farshi, T., 2021. Battle royale optimization algorithm. Neural Computing and Applications, 33(4),
        pp.1139-1157.
    """
    def __init__(self, config: BattleRoyaleOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__dyn_delta: float | None = None
        self.__lb_updated: np.ndarray | None = None
        self.__ub_updated: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = BattleRoyaleOptimizationConfig(**parameters)

    def _init_agent(self, position: list[Any] | np.ndarray | None = None, damage: int | None = None) -> Soldier:
        agent = super()._init_agent(position=position)
        soldier = Soldier(**agent.model_dump())
        if damage is not None:
            soldier.damage = damage
        return soldier

    def before_initialization(self):
        self.__lb_updated, self.__ub_updated = self._task.get_bounds()
        self.__dyn_delta = np.round(self._config.max_cycles / np.ceil(np.log10(self._config.max_cycles)))

    def optimization_step(self):            
        def find_idx_min_distance(target_pos: np.ndarray):
            target_pos = np.reshape(target_pos, (1, -1))
            dist_list = cdist(list_pos, target_pos, 'euclidean')
            dist_list = np.reshape(dist_list, (-1))
            k_zero = np.count_nonzero(dist_list == 0)
            if k_zero == len(dist_list):
                return np.random.choice(range(0, k_zero))
            return np.where(dist_list == np.min(dist_list[dist_list != 0]))[0][0]
        
        def evolve(soldier: Soldier) -> Soldier:
            pos = np.array(soldier.position)
            # Compare ith soldier with nearest one (jth)
            jdx = find_idx_min_distance(pos)
            pos_jdx = np.array(self._population[jdx].position)
            if soldier.cost < self._population[jdx].cost:
                # update Winner based on global best solution
                new_soldier = self._init_agent(
                    position=pos + np.random.normal(0, 1) * np.mean(np.array([pos, best_pos]), axis=0),
                    damage=soldier.damage - 1  # subtract damaged hurt -1 to go next battle
                )
                # Update Loser
                if self._population[jdx].damage < threshold:  # if loser not dead yet, move it based on general
                    pos_jdx_new = np.maximum(pos_jdx, best_pos) + np.random.uniform() * (
                        np.maximum(pos_jdx, best_pos) - np.minimum(pos_jdx, best_pos)
                    )
                    dam_jdx_new = self._population[jdx].damage + 1
                else:  # loser dead and respawn again
                    pos_jdx_new = np.random.uniform(lb, ub)
                    dam_jdx_new = 0
            else:
                # update Loser by following position of Winner
                new_soldier = self._population[jdx].copy()
                pos_jdx_new = pos_jdx + np.random.uniform() * (best_pos - pos_jdx)
                dam_jdx_new = 0
            # update the position of jdx soldier:
            # - when it is a loser (soldier.cost < self._population[jdx].cost), by moving based on General (if not dead
            #   yet) or by respawning again (if dead)
            # - when it is a winner (soldier.cost >= self._population[jdx].cost), by following the position of General
            #   to protect the King and General
            self._population[jdx] = self._init_agent(position=pos_jdx_new, damage=dam_jdx_new)
            return self._greedy_select_agent(soldier, new_soldier)
        
        cycle = self._current_cycle
        threshold = self._config.threshold
        lb, ub = self.__lb_updated, self.__ub_updated

        list_pos = np.array([soldier.position for soldier in self._population])
        best_pos = np.array(self._best_agent.position)
        
        self._population = [evolve(soldier) for soldier in self._population]

        if cycle >= self.__dyn_delta:
            pos_list = np.array([soldier.position for soldier in self._population])
            pos_std = np.std(pos_list, axis=0)
            lb = best_pos - pos_std
            ub = best_pos + pos_std
            self.__lb_updated = np.clip(lb, self.__lb_updated, self.__ub_updated)
            self.__ub_updated = np.clip(ub, self.__lb_updated, self.__ub_updated)
            self.__dyn_delta += np.round(self.__dyn_delta / 2)
