from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Virus(Agent):
    pass


class VirusColonySearchOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Virus Colony Search Optimization algorithm.
        lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5.\n
        sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor.
    """
    lamda: float
    sigma: float

    @field_validator("lamda")
    def correct_lamda(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"lamda\" must be a float in (0, 1.0). Got {v}")
        return v

    @field_validator("sigma")
    def correct_sigma(cls, v):
        if not 0 < v < 5:
            raise ValueError(f"\"sigma\" must be a float in (0, 5.0). Got {v}")
        return v
