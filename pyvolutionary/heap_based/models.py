from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Heap(Agent):
    pass


class HeapBasedOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Heap Based Optimization algorithm.
        degree (int): [2, 4], the degree level in Corporate Rank Hierarchy (CRH).
    """
    degree: int

    @field_validator("degree")
    def degree_validator(cls, v):
        if not 2 <= v <= 4:
            raise ValueError(f"Degree must be between 2 and 4. Got: {v}")
        return v
