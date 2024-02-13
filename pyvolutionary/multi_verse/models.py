from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Universe(Agent):
    pass


class MultiverseOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Multi-verse OptimizationC algorithm.
        wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper).\n
        wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper).
    """
    wep_min: float
    wep_max: float

    @field_validator("wep_min")
    def correct_wep_min(cls, v):
        if not 0.05 <= v <= 0.3:
            raise ValueError(f"\"wep_min\" must be a float in [0.05, 0.3]. Got {v}")
        return v

    @field_validator("wep_max")
    def correct_wep_max(cls, v):
        if not 0.75 <= v <= 1.0:
            raise ValueError(f"\"wep_max\" must be a float in [0.75, 1.0]. Got {v}")
        return v
