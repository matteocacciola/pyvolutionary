import math
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
    def __init__(self, config: ImperialistCompetitiveOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__countries: list[Country] = []
        self.__empires: list[EmpireClass] = []

    def _init_population(self):
        # Create countries
        k = self._config.number_of_countries
        countries = []
        i = 0
        while i < k:
            candidate = Country(self._init_agent())
            if (True for elem in countries if np.array_equal(elem.representation, candidate.representation)):
                countries.append(candidate)
                i += 1

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
        p = np.exp(-np.multiply(self._config.alpha_rate, empires_costs) / np.max(empires_costs))
        p = p / np.sum(p)
        for country in candidate_colonies:
            k = random_selection(p)
            self.__empires[k].add_colony(country)

        task_type = self._task.minmax
        self._population = [Transformer.transform(empire, task_type) for empire in self.__empires]

    def __assimilate(self):
        """
        Assimilate colonies into empires. The assimilation rate is defined by the user. The assimilation is done by
        replacing the representation of the colony with the representation of the empire.
        """
        assimilation_rate = self._config.assimilation_rate
        dim = self._task.space_dimension
        for empire in self.__empires:
            empire_representation = empire.emperor.representation
            for colony in empire.colonies:
                candidates = np.random.choice(
                    range(0, dim), int(np.round(dim * assimilation_rate, decimals=0)), replace=False,
                )
                colony.set_representation(self._init_agent(
                    [r if i in candidates else colony.representation[i] for i, r in enumerate(empire_representation)]
                ))

    def __revolution(self):
        """
        Revolution is a process that is applied to a colony. The revolution rate is defined by the user. The revolution
        is done by exchanging the representation of the colony with the representation of another colony.
        """
        revolution_probability = self._config.revolution_probability
        revolution_rate = self._config.revolution_rate
        fcn = self._fcn
        dim = self._task.space_dimension
        for empire in self.__empires:
            for i, colony in enumerate(empire.colonies):
                if np.random.random() <= revolution_probability:
                    # select the colony to exchange with
                    colony_representation = colony.representation
                    old_cost = colony.cost
                    number_of_tasks = int(math.ceil(revolution_rate * dim))

                    candidates = np.random.choice(range(0, dim), number_of_tasks, replace=False)
                    exchange = list(range(0, dim))

                    # remove the candidates from the exchange list
                    for index in candidates:
                        del exchange[index]

                    # select the candidates to exchange with
                    exchange_candidates = np.random.choice(exchange, number_of_tasks)
                    new_colony_representation = colony_representation
                    # exchange the candidates
                    for (x, y) in zip(candidates, exchange_candidates):
                        new_colony_representation[x], new_colony_representation[y] = (
                            colony_representation[y], colony_representation[x]
                        )
                    new_colony = Country(self._init_agent(new_colony_representation))
                    if new_colony.cost < old_cost:
                        empire.replace_colony(i, new_colony)

    def __inter_empire_war(self):
        """
        Inter-empire competition is a process that is applied to empires. The weakest empire is selected and a war is
        held between the weakest empire and the other empires. The probability of winning the war is proportional to
        the cost of the empire. The winning empire assimilates the weakest empire's colonies and the weakest colony
        from the weakest empire. If the weakest empire has no colonies, then the weakest emperor is assimilated by the
        winning empire.
        """
        if len(self.__empires) == 1:
            return

        total_cost = np.array([empire.cost for empire in self.__empires])

        # the weakest empire is the one with the highest cost
        weakest_empire_index = np.argmax(total_cost)
        weakest_empire = self.__empires[weakest_empire_index]
        p = np.exp(-np.multiply(self._config.alpha_rate, total_cost) / np.max(total_cost))

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

    def __intra_empire_war(self):
        """
        Intra-empire competition is a process that is applied to empires. It replaces a weakest emperor with the
        strongest colony in case of internal war. The probability of winning the war is proportional to the cost of the
        colony. Total cost remains unchanged for the empire
        """
        for empire in self.__empires:
            strongest_colony_index, strongest_colony = empire.get_strongest_colony()
            # if there is a picked colony and its cost is lower than the emperor, then swap them
            if strongest_colony and strongest_colony.cost < empire.emperor.cost:
                empire.replace_colony(strongest_colony_index, empire.emperor)
                empire.replace_emperor(strongest_colony)

    def optimization_step(self):
        # assimilation
        self.__assimilate()

        # revolution
        self.__revolution()

        # Intra - empire competition
        self.__intra_empire_war()

        # Inter - empire competition
        self.__inter_empire_war()

        task_type = self._task.minmax
        self._population = [Transformer.transform(empire, task_type) for empire in self.__empires]
