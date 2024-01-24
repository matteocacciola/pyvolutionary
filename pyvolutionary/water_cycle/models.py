from pydantic import conlist, field_validator

from ..models import Agent, BaseOptimizationConfig


class Stream(Agent):
    pass


class WaterCycleOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Water Cycle Optimization algorithm.
        nsr (int): [4, 10], Number of rivers + sea (sea = 1).\n
        wc (float): [1.0, 3.0], Weighting coefficient (C in the paper).\n
    """
    nsr: int
    wc: float

    @field_validator("nsr")
    def correct_nsr(cls, v):
        if not 4 <= v <= 10:
            raise ValueError(f"\"c1\" must be an integer in [4, 10]. Got {v}")
        return v

    @field_validator("wc")
    def correct_wc(cls, v):
        if not 1.0 <= v <= 3.0:
            raise ValueError(f"\"wc\" must be a float in [1.0, 3.0]. Got {v}")
        return v
