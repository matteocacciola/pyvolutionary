from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Soldier(Agent):
    wl: float | None = 2.0
    wg: float | None = 1.0

    def __init__(self, **kwargs):
        kwargs["wl"] = kwargs.get("wl") or 2.0
        kwargs["wg"] = kwargs.get("wg") or 1.0
        super().__init__(**kwargs)


class WarStrategyOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the War Strategy Optimization algorithm.
        rr (float): [0.1, 0.9], the probability of switching position updating.
    """
    rr: float

    @field_validator("rr")
    def correct_rr(cls, v):
        if not 0.1 <= v <= 0.9:
            raise ValueError(f"\"rr\" must be a float in [0.1, 0.9]. Got {v}")
        return v
