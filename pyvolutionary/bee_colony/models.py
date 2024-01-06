from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Bee(Agent):
    trials: int = 0


class BeeColonyOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Bee Colony Optimization algorithm.
        scouting_limit (int): [1, +Inf), the number of times a bee can scout before it is considered exhausted and is
        replaced.
    """
    scouting_limit: int

    @field_validator("scouting_limit")
    def correct_number_of_scouts(cls, v):
        if v < 1:
            raise ValueError(f"\"scouting_limit\" must be an integer greater than 1. Got {v}")
        return v
