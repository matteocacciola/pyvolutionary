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
    def __init__(self, config: FishSchoolSearchOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__school_weight = self._config.population_size * self._config.w_scale / 2.0
        self.__step_individual: np.ndarray | None = None
        self.__step_volitive: np.ndarray | None = None

    def before_initialization(self):
        self.__step_individual = self._config.step_individual_init * self._bandwidth()
        self.__step_volitive = self._config.step_volitive_init * self._bandwidth()

    def _init_agent(self, position: list[float] | np.ndarray | None = None) -> Fish:
        agent = super()._init_agent(position)
        return Fish(**agent.model_dump(), weight=self._config.w_scale / 2.0)

    def optimization_step(self):
        def move_individual(fish: Fish) -> Fish:
            pos = np.array(fish.position)
            new_fish = self._init_agent(pos + (si * self._uniform_position()))
            if new_fish.cost < fish.cost:
                delta_cost = abs(new_fish.cost - fish.cost)
                delta_pos = (np.array(new_fish.position) - pos).tolist()
                fish = new_fish
                fish.delta_cost = delta_cost
                fish.delta_pos = delta_pos
                return fish
            fish.delta_pos = np.zeros(sd).tolist()
            fish.delta_cost = 0
            return fish

        def update(init: float, final: float) -> np.ndarray:
            return np.full(
                self._task.space_dimension, init - (self._current_cycle + 1) * (init - final) / self._config.max_cycles
            )

        def feeding(fish: Fish) -> Fish:
            if max_delta_cost:
                fish.weight += (fish.delta_cost / max_delta_cost)
            fish.weight = np.clip(fish.weight, self._config.min_w, self._config.w_scale)
            return fish

        def volitive_movement(fish: Fish) -> Fish:
            pos = np.array(fish.position)
            new_pos = pos + (multiplier * (pos - barycenter) * sv * np.random.uniform(0, 1, sd))
            return self._init_agent(new_pos)

        # individual movement
        sd, si, sv = self._task.space_dimension, self.__step_individual, self.__step_volitive
        self._population = [move_individual(fish) for fish in self._population]

        # feeding
        max_delta_cost = max([fish.delta_cost for fish in self._population])
        self._population = [feeding(fish) for fish in self._population]

        # collective movements
        delta = sum([fish.delta_cost * np.array(fish.delta_pos) for fish in self._population], start=np.zeros(sd))
        density = sum([f.delta_cost for f in self._population])
        if density != 0:
            delta /= density
        self._population = [self._init_agent((np.array(fish.position) + delta).tolist()) for fish in self._population]

        # collective volitive movements
        school_weight = sum([fish.weight for fish in self._population])
        multiplier = -1 if school_weight > self.__school_weight else 1
        barycenter = (
            sum([np.array(fish.position) * fish.weight for fish in self._population], start=np.zeros(sd))
        ) / sum([fish.weight for fish in self._population])
        self._population = [volitive_movement(fish) for fish in self._population]

        # update steps
        self.__step_individual = update(self._config.step_individual_init, self._config.step_individual_final)
        self.__step_volitive = update(self._config.step_volitive_init, self._config.step_volitive_final)