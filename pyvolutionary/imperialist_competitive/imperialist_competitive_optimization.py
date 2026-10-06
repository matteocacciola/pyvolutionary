import math
from typing import Any
import numpy as np

from .classes import Country
from ..helpers import (
    random_selection,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .classes import Empire as EmpireClass, Transformer
from .models import ImperialistCompetitiveOptimizationConfig


class ImperialistCompetitiveOptimization(OptimizationAbstract):
    """
    Implementation of the Imperialist Competitive Optimization algorithm.

    Args:
        config (ParticleSwarmOptimizationConfig): an instance of ParticleSwarmOptimizationConfig class.
            {parse_obj_doc(ParticleSwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Esmaeilzadeh, E., & Ghane, M. (2013). Imperialist competitive algorithm: A metaheuristic algorithm for
        optimization inspired by imperialistic competition. 2013 3rd International Conference on Computer and Knowledge
        Engineering (ICCKE), 1–6. https://doi.org/10.1109/ICCKE.2013.6687751
    [2] Esmaeilzadeh, E., & Ghane, M. (2014). Imperialist competitive algorithm: An algorithm for optimization inspired
        by imperialistic competition. Applied Soft Computing Journal, 14, 240–256.
        https://doi.org/10.1016/j.asoc.2013.08.006
    """
    def __init__(self, config: ImperialistCompetitiveOptimizationConfig | None = None, debug: bool | None = False):
        super().__init__(config, debug)
        self.__countries: list[Country] = []
        self.__empires: list[EmpireClass] = []

    def set_config_parameters(self, parameters: dict[str, Any]):
        self._config = ImperialistCompetitiveOptimizationConfig(**parameters)

    def _init_population(self):
        # the empires of a previous run are discarded
        self.__empires = []

        # Create countries
        k = self._config.number_of_countries
        countries = []
        # a candidate is discarded when another country has the same representation; a small (e.g. discrete) search
        # space may have fewer distinct representations than k, so duplicates are accepted after many attempts
        max_attempts = 100 * k
        attempts = 0
        while len(countries) < k:
            candidate = Country(self._init_agent())
            attempts += 1
            is_duplicate = any(np.array_equal(elem.representation, candidate.representation) for elem in countries)
            if not is_duplicate or attempts > max_attempts:
                countries.append(candidate)

        self.__countries = countries

        # Create empires
        costs = np.array([np.sum(countries[i].cost) for i in range(0, len(countries))])
        indices = np.argsort(costs)
        new_countries = np.array([countries[i] for i in indices])

        candidate_empires = new_countries[:self._config.population_size]
        candidate_colonies = new_countries[self._config.population_size:]

        for ctr in candidate_empires:
            self.__empires.append(EmpireClass(ctr))

        empires_costs = np.array([np.sum(empire.cost) for empire in self.__empires])
        p = self._power_weights(empires_costs)
        p = p / np.sum(p)
        for country in candidate_colonies:
            k = random_selection(p)
            self.__empires[k].add_colony(country)

        task_type = self._task.minmax
        self._population = [Transformer.transform(empire, task_type) for empire in self.__empires]

    def _power_weights(self, costs: np.ndarray) -> np.ndarray:
        """
        The (not normalized) weights of the empires, i.e. their power: the lower the cost, the higher the weight.
        The costs are scaled by their largest magnitude: dividing them by the largest cost would invert the weights
        (and overflow) with negative costs, e.g. of maximization tasks; with positive costs, it is the same.
        :param costs: the costs of the empires
        :return: the weights of the empires
        :rtype: np.ndarray
        """
        return np.exp(-np.multiply(self._config.alpha_rate, costs) / np.max(np.abs(costs)))

    def __inter_empire_war__(self):
        """
        Inter-empire competition is a process that is applied to empires. The weakest empire is selected and a war is
        held between the weakest empire and the other empires. The probability of winning the war is proportional to
        the cost of the empire. The winning empire assimilates the weakest empire's colonies and the weakest colony
        from the weakest empire. If the weakest empire has no colonies, then the weakest emperor is assimilated by the
        winning empire.
        """
        total_cost = np.array([empire.cost for empire in self.__empires])

        # the weakest empire is the one with the highest cost
        weakest_empire_index = np.argmax(total_cost)
        weakest_empire = self.__empires[weakest_empire_index]
        p = self._power_weights(total_cost)

        # the weakest empire has a probability of 0 to win the war
        p[weakest_empire_index] = 0
        p = p / np.sum(p)
        # if all probabilities are 0, then the weakest empire wins the war
        if np.any(np.isnan(p)):
            p[np.isnan(p)] = 0
            if all(p == 0):
                p[:] = 1
            p = p / sum(p)

        # if the weakest empire has colonies, then select the weakest colony and add it to the winning empire
        if weakest_empire.number_of_colonies > 0:
            weakest_empire_colonies_cost = np.array([colony.cost for colony in weakest_empire.colonies])
            weakest_colony_index = np.argmax(weakest_empire_colonies_cost)
            weakest_colony = weakest_empire.get_colony(weakest_colony_index)

            winning_empire_index = random_selection(p)
            winning_empire = self.__empires[winning_empire_index]

            winning_empire.add_colony(weakest_colony)
            weakest_empire.delete_colony(weakest_colony_index)

        # if the weakest empire has no colonies, then select the weakest emperor and add it to the winning empire
        if weakest_empire.number_of_colonies == 0:
            winning_empire_index = random_selection(p)
            winning_empire = self.__empires[winning_empire_index]

            winning_empire.add_colony(weakest_empire.emperor)
            del self.__empires[self.__empires.index(weakest_empire)]

    def optimization_step(self):
        def assimilate_colonies(empire: EmpireClass) -> EmpireClass:
            empire_representation = np.array(empire.emperor.representation)
            for colony in empire.colonies:
                candidates = np.random.choice(dim, n_assimilated, replace=False)
                # the colony takes the coordinates of the emperor in the candidate dimensions
                is_candidate = np.zeros(dim, dtype=bool)
                is_candidate[candidates] = True
                colony.set_representation(self._init_agent(
                    np.where(is_candidate, empire_representation, colony.representation)
                ))
            return empire

        def revolution(empire: EmpireClass) -> EmpireClass:
            for i, colony in enumerate(empire.colonies):
                # with a single dimension there is nothing to exchange
                if dim > 1 and np.random.random() <= revolution_probability:
                    old_cost = colony.cost
                    # at least one dimension must be left out of the candidates, to exchange with
                    number_of_tasks = min(int(math.ceil(revolution_rate * dim)), dim - 1)
                    candidates = np.random.choice(dim, number_of_tasks, replace=False)
                    # the dimensions to exchange with are the ones that are not candidates
                    exchange = [index for index in range(0, dim) if index not in candidates]
                    exchange_candidates = np.random.choice(exchange, number_of_tasks)
                    # exchange the candidates on a copy: the colony is kept as it is if the new one is not better
                    new_colony_representation = list(colony.representation)
                    for (x, y) in zip(candidates, exchange_candidates):
                        new_colony_representation[x], new_colony_representation[y] = (
                            new_colony_representation[y], new_colony_representation[x]
                        )
                    new_colony = Country(self._init_agent(new_colony_representation))
                    if new_colony.cost < old_cost:
                        empire.replace_colony(i, new_colony)
            return empire

        def intra_empire_war(empire: EmpireClass) -> EmpireClass:
            strongest_colony_index, strongest_colony = empire.get_strongest_colony()
            # if there is a picked colony and its cost is lower than the emperor, then swap them
            if strongest_colony and strongest_colony.cost < empire.emperor.cost:
                empire.replace_colony(strongest_colony_index, empire.emperor)
                empire.replace_emperor(strongest_colony)
            return empire

        assimilation_rate = self._config.assimilation_rate
        revolution_probability = self._config.revolution_probability
        revolution_rate = self._config.revolution_rate
        dim = self._task.space_dimension
        n_assimilated = int(np.round(dim * assimilation_rate, decimals=0))

        # assimilation
        self.__empires = [assimilate_colonies(empire) for empire in self.__empires]

        # revolution
        self.__empires = [revolution(empire) for empire in self.__empires]

        # Intra - empire competition
        self.__empires = [intra_empire_war(empire) for empire in self.__empires]

        # Inter - empire competition
        if len(self.__empires) > 1:
            self.__inter_empire_war__()

        task_type = self._task.minmax
        self._population = [Transformer.transform(empire, task_type) for empire in self.__empires]
