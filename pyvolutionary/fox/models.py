from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Fox(Agent):
    pass


class FoxOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Fox Optimization algorithm.
        c1 (float): [0, 0.18], the probability of jumping (c1 in the paper)
        c2 (float): [0.19, 1], the probability of jumping (c2 in the paper)
    """
    c1: float
    c2: float

    @field_validator("c1")
    def correct_c1(cls, v):
        if not 0 <= v <= 0.18:
            raise ValueError("c1 must be in [0, 0.18]")
        return v

    @field_validator("c2")
    def correct_c2(cls, v):
        if not 0.19 <= v <= 1:
            raise ValueError("c2 must be in [0.19, 1]")
        return v
