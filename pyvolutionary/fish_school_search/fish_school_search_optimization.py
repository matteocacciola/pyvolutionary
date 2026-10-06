from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Fish, FishSchoolSearchOptimizationConfig


class FishSchoolSearchOptimization(OptimizationAbstract):
    """
    Implementation of the Fish School Search Optimization.

    Args:
        config (FishSchoolSearchOptimizationConfig): an instance of FishSchoolSearchOptimizationConfig class.
            {parse_obj_doc(FishSchoolSearchOptimizationConfig)}

    Bibliography
    ----------
    [1] Bastos Filho, Lima Neto, Lins, D. O. Nascimento and P. Lima, A novel search algorithm based on fish school
        behavior, in 2008 IEEE International Conference on Systems, Man and Cybernetics, Oct 2008, pp. 2646–2651.
    """
    def __init__(self, config: FishSchoolSearchOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__school_weight: float | None = None
        self.__step_individual: np.ndarray | None = None
        self.__step_volitive: np.ndarray | None = None

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = FishSchoolSearchOptimizationConfig(**parameters)

    def before_initialization(self):
        self.__school_weight = self._config.population_size * self._config.w_scale / 2.0
        self.__step_individual = self._config.step_individual_init * self._task.bandwidth()
        self.__step_volitive = self._config.step_volitive_init * self._task.bandwidth()

    def _init_agent(self, position: list[Any] | np.ndarray | None = None, weight: float | None = None) -> Fish:
        agent = super()._init_agent(position)
        # a fish keeps its weight when it moves; a new fish starts from half of the weight scale
        return Fish(**agent.__dict__, weight=weight if weight is not None else self._config.w_scale / 2.0)

    def optimization_step(self):
        def move_individual(fish: Fish) -> Fish:
            # a random step in [-1, 1] per dimension, scaled by the individual step: kept only if it improves the fish
            pos = np.array(fish.position)
            new_fish = self._init_agent(pos + si * np.random.uniform(-1, 1, sd), fish.weight)
            if new_fish.cost < fish.cost:
                new_fish.delta_cost = fish.cost - new_fish.cost
                new_fish.delta_pos = (np.array(new_fish.position) - pos).tolist()
                return new_fish
            fish.delta_pos = np.zeros(sd).tolist()
            fish.delta_cost = 0
            return fish

        def feeding(fish: Fish) -> Fish:
            if max_delta_cost:
                fish.weight += (fish.delta_cost / max_delta_cost)
            fish.weight = float(np.clip(fish.weight, self._config.min_w, self._config.w_scale))
            return fish

        def volitive_movement(fish: Fish) -> Fish:
            # towards the barycenter if the school gained weight (contraction), away from it otherwise (dilation)
            pos = np.array(fish.position)
            direction = pos - barycenter
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            new_pos = pos + multiplier * sv * np.random.uniform(0, 1, sd) * direction
            return self._init_agent(new_pos, fish.weight)

        def step(init: float, final: float) -> np.ndarray:
            # the steps decrease linearly, from init to final, relative to the bandwidth of the search space
            return (init - self._current_cycle * (init - final) / self._config.max_cycles) * self._task.bandwidth()

        # individual movement
        sd, si, sv = self._task.space_dimension, self.__step_individual, self.__step_volitive
        self._population = [move_individual(fish) for fish in self._population]

        # feeding
        max_delta_cost = max([fish.delta_cost for fish in self._population])
        self._population = [feeding(fish) for fish in self._population]

        # collective-instinctive movement: the school moves along the improvements, weighted by their size
        delta = sum([fish.delta_cost * np.array(fish.delta_pos) for fish in self._population], start=np.zeros(sd))
        density = sum([f.delta_cost for f in self._population])
        if density != 0:
            delta /= density
        self._population = [
            self._init_agent(np.array(fish.position) + delta, fish.weight) for fish in self._population
        ]

        # collective-volitive movement
        school_weight = sum([fish.weight for fish in self._population])
        multiplier = -1 if school_weight > self.__school_weight else 1
        self.__school_weight = school_weight
        barycenter = (
            sum([np.array(fish.position) * fish.weight for fish in self._population], start=np.zeros(sd))
        ) / school_weight
        self._population = [volitive_movement(fish) for fish in self._population]

        # update steps
        self.__step_individual = step(self._config.step_individual_init, self._config.step_individual_final)
        self.__step_volitive = step(self._config.step_volitive_init, self._config.step_volitive_final)
