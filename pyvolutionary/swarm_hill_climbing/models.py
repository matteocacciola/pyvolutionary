from pydantic import model_validator

from ..models import Agent, BaseOptimizationConfig


class Climber(Agent):
    pass


class SwarmHillClimbingOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Swarm Hill Climbing Optimization algorithm.
        neighbour_size (int): [2, int(pop_size/2)], number of neighbours to be generated for each agent.
    """
    neighbour_size: int

    @model_validator(mode="after")
    def validate_n_elites(self) -> "SwarmHillClimbingOptimizationConfig":
        if not 2 <= self.neighbour_size <= int(self.population_size / 2):
            raise ValueError("Number of neighbours must be in [2, int(pop_size/2)]")
        return self
