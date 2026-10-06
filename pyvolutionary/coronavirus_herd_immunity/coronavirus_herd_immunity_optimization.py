from typing import Any
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Patient, CoronavirusHerdImmunityOptimizationConfig


class CoronavirusHerdImmunityOptimization(OptimizationAbstract):
    """
    Implementation of the Coronavirus Herd Immunity Optimization algorithm.

    Args:
        config (CoronavirusHerdImmunityOptimizationConfig): an instance of CoronavirusHerdImmunityOptimizationConfig
            class.
            {parse_obj_doc(CoronavirusHerdImmunityOptimizationConfig)}

    Bibliography
    ----------
    [1] Al-Betar, M.A., Alyasseri, Z.A.A., Awadallah, M.A. et al. Coronavirus herd immunity optimizer (CHIO). Neural
        Comput & Applic 33, 5011–5042 (2021).
    """

    def __init__(self, config: CoronavirusHerdImmunityOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = CoronavirusHerdImmunityOptimizationConfig(**parameters)

    def _init_agent(
        self, position: list[Any] | np.ndarray | None = None, status: int | None = 0, age: int | None = 0
    ) -> Patient:
        agent = super()._init_agent(position)

        return Patient(**agent.__dict__, status=status, age=age)

    def after_initialization(self):
        infected = int(self._config.C0 * self._config.population_size)
        idx_infected = np.random.choice(self._config.population_size, infected, replace=False)
        self._population = [agent.model_copy(
            update={"status": 1}
        ) if idx in idx_infected else agent for idx, agent in enumerate(self._population)]

    def optimization_step(self):
        def select_random_index(k: int) -> int:
            rand = np.random.uniform()
            ratio = (1.0 / 3) * brr
            if rand < ratio and len(confirmed) > 0:
                # the current patient gets in touch with a confirmed one, so it may get infected
                is_corona_list[k] = True
                return np.random.choice(confirmed)
            if ratio <= rand < 2 * ratio and len(normal) > 0:
                return np.random.choice(normal)
            if 2 * ratio <= rand < brr and len(recovered) > 0:
                cost_list_candidates = [self._population[idx].cost for idx in recovered]
                return recovered[np.argmin(cost_list_candidates)]  # Found the index of best cost
            return k

        def evolve(idx: int, patient: Patient) -> Patient:
            pos = np.array(patient.position)
            idxs = [select_random_index(idx) for _ in range(0, n_dims)]
            pos_selected = [self._population[idxs[jdx]].position[jdx] for jdx in range(0, n_dims)]
            pos_new = pos + np.random.uniform(0, 1, n_dims) * (pos - np.array(pos_selected))
            new_agent = self._init_agent(pos_new, status=patient.status, age=patient.age)
            p = patient.model_copy(update={"age": patient.age + 1})
            return self._greedy_select_agent(p, new_agent)

        def update_immunities(idx: int, patient: Patient) -> Patient:
            delta_fx = np.mean(cost_list)  # Calculate immunity mean of population
            # change the solution from normal to confirmed
            if patient.cost < delta_fx and patient.status == 0 and is_corona_list[idx]:
                patient.status = 1
                patient.age = 1
            # change the solution from confirmed to recovered
            if delta_fx < patient.cost and patient.status == 1:
                patient.status = 2
                patient.age = 0
            # fatality: a confirmed patient not improving for max_age cycles is regenerated from scratch
            if patient.age >= max_age and patient.status == 1:
                patient = self._init_agent()
            return patient

        brr = self._config.brr
        max_age = self._config.max_age

        pop_size = self._config.population_size
        n_dims = self._task.space_dimension
        is_corona_list = [False, ] * pop_size

        normal = [idx for idx, patient in enumerate(self._population) if patient.status == 0]
        confirmed = [idx for idx, patient in enumerate(self._population) if patient.status == 1]
        recovered = [idx for idx, patient in enumerate(self._population) if patient.status == 2]

        self._population = [evolve(idx, patient) for idx, patient in enumerate(self._population)]

        cost_list = np.array([agent.cost for agent in self._population])
        self._population = [update_immunities(idx, patient) for idx, patient in enumerate(self._population)]
