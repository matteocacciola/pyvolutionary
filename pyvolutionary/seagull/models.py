from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Seagull(Agent):
    pass


class SeagullOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for the Seagull Optimization algorithm.
        fc (float): [1.0, 10.0] -> frequency of employing variable A (A linear decreased from fc to 0)
    """
    fc: float

    @field_validator("fc")
    def fc_validator(cls, v):
        if not (1.0 <= v <= 10.0):
            raise ValueError("fc must be between 1.0 and 10.0")
        return v
