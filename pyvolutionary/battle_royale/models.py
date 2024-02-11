from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Soldier(Agent):
    damage: int = 0


class BattleRoyaleOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Battle Royal Optimization algorithm.
        threshold (int): [2, 5], dead threshold.
    """
    threshold: int

    @field_validator("threshold")
    def threshold_validator(cls, v):
        if not 2 <= v <= 5:
            raise ValueError(f"Threshold must be between 2 and 5. Got: {v}")
        return v
