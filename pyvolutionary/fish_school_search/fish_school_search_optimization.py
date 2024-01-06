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

    def __update_steps__(self):
        """
        Update the steps of the fishes. The steps are updated according to the current cycle. The steps are updated
        according to the following equations: \n
        step_individual = step_individual_init - (cycles + 1) * (step_individual_init - step_individual_final) / max_cycles \n
        step_volitive = step_volitive_init - (cycles + 1) * (step_volitive_init - step_volitive_final) / max_cycles \n
        where: \n
        step_individual_init: length of initial individual step \n
        step_individual_final: length of final individual step \n
        step_volitive_init: length of initial volatile step \n
        step_volitive_final: length of final volatile step \n
        cycles: current cycle \n
        max_cycles: maximum number of cycles.
        """
        def update(init: float, final: float) -> np.ndarray:
            return np.full(
                self._task.space_dimension, init - (self._cycles + 1) * (init - final) / self._config.max_cycles
            )

        self.__step_individual = update(self._config.step_individual_init, self._config.step_individual_final)
        self.__step_volitive = update(self._config.step_volitive_init, self._config.step_volitive_final)

    def __feeding__(self):
        """
        Feed all fishes. The feeding is done according to the following equation: \n
        weight = weight + delta_cost / max_delta_cost \n
        where: \n
        weight: weight of the fish \n
        delta_cost: difference between the current cost and the previous cost of the fish \n
        max_delta_cost: maximum difference between the current cost and the previous cost of all fishes.
        """
        def feeding(fish: Fish, mxd: float) -> Fish:
            if mxd:
                fish.weight += (fish.delta_cost / mxd)
            fish.weight = np.clip(fish.weight, self._config.min_w, self._config.w_scale)
            return fish

        max_delta_cost = max([fish.delta_cost for fish in self._population])
        self._population = [feeding(fish, max_delta_cost) for fish in self._population]

    def __individual_movement__(self):
        """
        Perform individual movement for each fish.
        """
        def move(fish: Fish, sd: int, si: np.ndarray) -> Fish:
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

        self._population = [move(
            fish, self._task.space_dimension, self.__step_individual
        ) for fish in self._population]

    def __collective_instinctive_movement__(self):
        """
        Perform collective instinctive movement.
        """
        cost_eval_enhanced = sum(
            [fish.delta_cost * np.array(fish.delta_pos) for fish in self._population],
            start=np.zeros(self._task.space_dimension)
        )
        density = sum([f.delta_cost for f in self._population])
        if density != 0:
            cost_eval_enhanced /= density
        self._population = [
            self._init_agent((np.array(fish.position) + cost_eval_enhanced).tolist()) for fish in self._population
        ]

    def __collective_volitive_movement__(self):
        """
        Perform collective volitive movement.
        """
        def move(fish: Fish, m: int, bc: np.ndarray, s: np.ndarray, d: int) -> Fish:
            pos = np.array(fish.position)
            step = (pos - bc) * s
            new_pos = pos + (m * step * np.random.uniform(0, 1, d))
            return self._init_agent(new_pos)

        school_weight = sum([fish.weight for fish in self._population])
        multiplier = -1 if school_weight > self.__school_weight else 1

        barycenter = sum(
            [np.array(fish.position) * fish.weight for fish in self._population],
            start=np.zeros(self._task.space_dimension)
        )
        barycenter /= sum([fish.weight for fish in self._population])

        self._population = [move(
            fish, multiplier, barycenter, self.__step_volitive, self._task.space_dimension
        ) for fish in self._population]

    def optimization_step(self):
        self.__individual_movement__()
        self.__feeding__()
        self.__collective_instinctive_movement__()
        self.__collective_volitive_movement__()
        self.__update_steps__()
