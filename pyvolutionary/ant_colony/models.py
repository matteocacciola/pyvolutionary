from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Ant(Agent):
    pass


class AntColonyOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Ant Colony Optimization algorithm.
        archive_size (int): the size of the archive of the colony.\n
        intent_factor (float): [0, 1), the intent factor of the colony.\n
        zeta (float): the zeta parameter of the colony.
    """
    archive_size: int
    intent_factor: float
    zeta: float

    @field_validator("intent_factor")
    def correct_intent_factor(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"intent_factor\" must be a positive float lower than 1. Got {v}")
        return v
