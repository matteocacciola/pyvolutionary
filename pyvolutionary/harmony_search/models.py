from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Harmony(Agent):
    pass


class HarmonySearchOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Harmony Search Optimization algorithm.
        consideration_rate (float): [0, 1), the rate of using harmony memory.\n
        pitch_adjusting_rate (float): [0, 1), the rate of pitch adjusting.
    """
    consideration_rate: float
    pitch_adjusting_rate: float

    @field_validator("consideration_rate")
    def correct_consideration_rate(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"consideration_rate\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("pitch_adjusting_rate")
    def correct_pitch_adjusting_rate(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"pitch_adjusting_rate\" must be a positive float lower than 1. Got {v}")
        return v
