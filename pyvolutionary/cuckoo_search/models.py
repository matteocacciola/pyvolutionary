from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Cuckoo(Agent):
    pass


class CuckooSearchOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Cuckoo Search Optimization algorithm.
        p_a (float): [0.1, 0.7], probability a.
    """
    p_a: float

    @field_validator("p_a")
    def correct_p_a(cls, v):
        if not 0.1 <= v <= 0.7:
            raise ValueError(f"\"p_a\" must be a float in [0.1, 0.7]. Got {v}")
        return v
