from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Grasshopper(Agent):
    pass


class GrasshopperOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Grasshopper Optimization Algorithm.
        c_min (float): [0.00001, 0.2], coefficient c min, default = 0.00004.\n
        c_max (float): [0.2, 5.0], coefficient c max, default = 2.0.
    """
    c_min: float
    c_max: float

    @field_validator("c_min")
    def validate_c_min(cls, v):
        if not 0.00001 <= v <= 0.2:
            raise ValueError("c_min must be in range [0.00001, 0.2].")
        return v

    @field_validator("c_max")
    def validate_c_max(cls, v):
        if not 0.2 <= v <= 5.0:
            raise ValueError("c_max must be in range [0.2, 5.0].")
        return v
