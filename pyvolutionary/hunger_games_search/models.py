from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Individual(Agent):
    hunger: float = 1.0


class HungerGamesSearchOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Hunger Games Search Optimization algorithm.
        PUP (float): [0.01, 0.2], The probability of updating position (L in the paper).\n
        LH (float): [1000, 20000], Largest hunger / threshold.
    """
    PUP: float
    LH: float

    @field_validator("PUP")
    def PUP_validator(cls, v):
        if not 0.01 <= v <= 0.2:
            raise ValueError(f"PUP must be in [0.01, 0.2]. Got: {v}")
        return v

    @field_validator("LH")
    def LH_validator(cls, v):
        if not 1000 <= v <= 20000:
            raise ValueError(f"LH must be an integer in [1000, 20000]. Got: {v}")
        return v
