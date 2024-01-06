import numpy as np

from .models import Empire as EmpireModel
from ..enums import TaskType
from ..helpers import calculate_fitness
from ..models import Agent


class Country:
    def __init__(self, agent: Agent):
        self.__colonies = []
        self.__from_agent(agent)

    def __from_agent(self, agent: Agent):
        self.__representation = agent.position
        self.__makespan = agent.cost

    @property
    def is_colony(self):
        return False

    @property
    def is_imperialist(self):
        return not self.is_colony

    @property
    def representation(self):
        return self.__representation

    @property
    def position(self):
        return self.representation

    @property
    def cost(self):
        return self.__makespan

    def set_representation(self, agent: Agent):
        self.__from_agent(agent)


class Empire:
    def __init__(self, emperor: Country):
        self.__emperor = emperor
        self.__colonies: list[Country] = []
        self.__cost = emperor.cost

    def __calculate_cost(self):
        self.__cost = self.__emperor.cost + sum([x.cost for x in self.__colonies])

    def replace_colony(self, index: int, colony: Country):
        self.__colonies[index] = colony
        self.__calculate_cost()

    def replace_emperor(self, colony: Country):
        self.__emperor = colony
        self.__calculate_cost()

    def delete_colony(self, index: int):
        del self.__colonies[index]
        self.__calculate_cost()

    @property
    def cost(self) -> float:
        return self.__cost

    def add_colony(self, colony: Country, index: int = 0):
        self.__colonies.insert(len(self.__colonies) if index == 0 else index, colony)
        self.__calculate_cost()

    def remove_colony(self, index: int):
        del self.__colonies[index]
        self.__calculate_cost()

    @property
    def number_of_colonies(self) -> int:
        return len(self.__colonies)

    @property
    def colonies(self) -> list[Country]:
        return self.__colonies

    def get_colony(self, index: int) -> Country:
        return self.__colonies[index]

    def get_strongest_colony(self) -> tuple[int, Country | None]:
        """
        Among all the colonies, find the one and its index with the lowest cost
        :return: a tuple of the index and the colony with the lowest cost
        :rtype: tuple[int, Country]
        """
        if self.number_of_colonies == 0:
            return -1, None

        strongest_colony_index = np.argmin(np.array([colony.cost for colony in self.__colonies]))
        strongest_colony = self.get_colony(strongest_colony_index)
        return strongest_colony_index, strongest_colony

    @property
    def emperor(self) -> Country:
        return self.__emperor


class Transformer:
    """
    This class is used to transform the Empire class to EmpireModel class.
    """
    @staticmethod
    def transform(empire: Empire, task_type: TaskType) -> EmpireModel:
        """
        :type empire: Empire
        :param empire: the empire to be transformed
        :param task_type: the type of the task, i.e. whether it is a minimization or maximization task
        :return: an instance of EmpireModel class, i.e. the pydantic representation of the empire
        """
        cost = empire.cost
        return EmpireModel(
            position=empire.emperor.representation, cost=cost, fitness=calculate_fitness(cost, task_type)
        )
