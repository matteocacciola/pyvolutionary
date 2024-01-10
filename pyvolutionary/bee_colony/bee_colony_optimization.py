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

    def optimization_step(self):
        def food_source_dance(idx: int, bee: Bee):
            # a randomly chosen solution is used in producing a mutant solution of the i-th solution
            # randomly selected solution must be different from the i-th solution
            partner_index = get_partner_index(idx, population_size)
            partner = self._population[partner_index]
            # generate a mutant solution by perturbing the current solution "index" with a random number
            pos_new = np.array(bee.position) + phi * (np.array(bee.position) - np.array(partner.position))
            return self._greedy_select_agent(bee, self._init_agent(pos_new))

        def send_onlooker_bees(idx: int) -> Bee:
            """
            Send onlooker bees to search for food sources. Each onlooker bee will dance on a food source. The probability of
            each onlooker bee to dance on a food source is proportional to the quality of the food source. The better the
            food source, the higher the probability of being selected. The probability of each food source is calculated
            using the following formula:
                p_i = cost_i / sum(costs)
            where p_i is the probability of the i-th food source, and cost_i is the cost of the i-th food source. The
            probability of each food source is calculated using the costs of the employed bees. The onlooker bees will use a
            roulette wheel selection to select a food source.
            """
            # Select an employed bee using roulette wheel selection
            selected_bee = self._population[roulette_wheel_index(probabilities)]
            return food_source_dance(idx, selected_bee)

        population_size = self._config.population_size
        dims = self._task.space_dimension
        phi = np.random.uniform(low=-1, high=1, size=dims)

        # generate and evaluate a neighbour point to every food source; it means to send the employed bees
        self._population = [food_source_dance(idx, bee) for idx, bee in enumerate(self._population)]

        # based to probability, generate a neighbour point and evaluate again some food sources
        # same food source can be evaluated multiple times
        employed_costs = np.array([agent.cost for agent in self._population])
        probabilities = employed_costs / np.sum(employed_costs)
        self._population = [send_onlooker_bees(idx) for idx in range(0, population_size)]

        # abandon the food sources which have not been improved after a predefined number of trials; it means to send
        # the scout bees
        scouting_limit = self._config.scouting_limit
        trials = np.array([food.trials for food in self._population])
        abandoned = np.where(trials >= scouting_limit)[0]
        self._population = [
            agent if idx not in abandoned else self._init_agent() for idx, agent in enumerate(self._population)
        ]
