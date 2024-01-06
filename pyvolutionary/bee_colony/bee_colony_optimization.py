import numpy as np

from ..helpers import (
    get_partner_index,
    roulette_wheel_index,
    parse_obj_doc,  # type: ignore
)
from ..abstract import OptimizationAbstract
from .models import Bee, BeeColonyOptimizationConfig


class BeeColonyOptimization(OptimizationAbstract):
    """
    Implementation of the Bee Colony Optimization algorithm.

    Args:
        config (BeeColonyOptimizationConfig): an instance of BeeColonyOptimizationConfig class.
            {parse_obj_doc(AntColonyOptimizationConfig)}

    Bibliography
    ----------
    [1] D. Karaboga, An idea based on honey bee swarm for numerical optimization, Technical Report TR06, Erciyes
        University, Engineering Faculty, Computer Engineering Department, 2005.
    [2] D. Karaboga, B. Basturk, On The Performance Of Artificial Bee Colony (ABC) Algorithm, Applied Soft Computing,
        8(1), 687-697, 2008.
    [3] D. Karaboga, B. Basturk, A Powerful And Efficient Algorithm For Numerical Function Optimization: Artificial Bee
        Colony (ABC) Algorithm, Journal of Global Optimization, 39(3), 459-471, 2007.
    """
    def __init__(self, config: BeeColonyOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self._config.population_size = int(self._config.population_size / 2)

    def _init_agent(self, position: list[float] | np.ndarray | None = None) -> Bee:
        agent = super()._init_agent(position)
        return Bee(**agent.model_dump())

    def _greedy_select_agent(self, agent: Bee, new_agent: Bee) -> Bee:
        """
        Perform the greedy selection between the current agent and the new one. The greedy selection is performed by
        comparing the costs of each agent. The one with the lowest cost is kept.
        :param agent: the current agent
        :param new_agent: the new agent
        :return: the best agent
        """
        # if the current agent can not be improved, increase its trial counter
        return new_agent if new_agent.cost < agent.cost else agent.model_copy(update={"trials": agent.trials + 1})

    def __send_employed_bees__(self) -> None:
        """
        Send employed bees to search for food sources. Each employed bee will dance on a food source.
        :return:
        """
        for i, bee in enumerate(self._population):
            self.__food_source_dance__(i, bee)

    def __food_source_dance__(self, index: int, bee: Bee):
        """
        Perform a food source dance. The dance is performed by generating a mutant solution and evaluating it. If the
        mutant solution is better than the current solution, the current solution is replaced with the mutant solution.
        Otherwise, the trial counter of the current solution is increased by one.
        :param index:
        :param bee:
        :return:
        """
        # a randomly chosen solution is used in producing a mutant solution of the i-th solution
        # randomly selected solution must be different from the i-th solution
        partner_index = get_partner_index(index, self._config.population_size)
        partner = self._population[partner_index]

        # generate a mutant solution by perturbing the current solution "index" with a random number
        phi = np.random.uniform(low=-1, high=1, size=self._task.space_dimension)
        pos_new = np.array(bee.position) + phi * (np.array(bee.position) - np.array(partner.position))
        self._population[index] = self._greedy_select_agent(bee, self._init_agent(pos_new))

    def __send_onlooker_bees__(self):
        """
        Send onlooker bees to search for food sources. Each onlooker bee will dance on a food source. The probability of
        each onlooker bee to dance on a food source is proportional to the quality of the food source. The better the
        food source, the higher the probability of being selected. The probability of each food source is calculated
        using the following formula:
            p_i = cost_i / sum(costs)
        where p_i is the probability of the i-th food source, and cost_i is the cost of the i-th food source. The
        probability of each food source is calculated using the costs of the employed bees. The onlooker bees will use a
        roulette wheel selection to select a food source.
        :return:
        """
        # Calculate the probabilities of each employed bee
        employed_costs = np.array([agent.cost for agent in self._population])
        probabilities = employed_costs / np.sum(employed_costs)
        for idx in range(0, self._config.population_size):
            # Select an employed bee using roulette wheel selection
            selected_bee = self._population[roulette_wheel_index(probabilities)]
            self.__food_source_dance__(idx, selected_bee)

    def __send_scout_bees__(self):
        """
        Send scout bees to search for food sources. If the number of trials of a food source exceeds a predefined limit,
        the food source is abandoned and a new food source is generated. The new food source is generated randomly.
        """
        trials = np.array([food.trials for food in self._population])

        # Check the number of trials for each employed bee and abandon the food sources if the limit is exceeded
        abandoned = np.where(trials >= self._config.scouting_limit)[0]
        for idx in abandoned:
            self._population[idx] = self._init_agent()  # replace food source with a brand new one

    def optimization_step(self):
        # generate and evaluate a neighbour point to every food source
        self.__send_employed_bees__()

        # based to probability, generate a neighbour point and evaluate again some food sources
        # same food source can be evaluated multiple times
        self.__send_onlooker_bees__()

        # abandon the food sources which have not been improved after a predefined number of trials
        self.__send_scout_bees__()
