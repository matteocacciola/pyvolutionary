from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Fox(Agent):
    pass


class FoxOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Fox Optimization algorithm.
        c1 (float): (-100., 100.), the probability of jumping (c1 in the paper).\n
        c2 (float): (-100., 100.), the probability of jumping (c2 in the paper),
        pp (float): (0.0, 1.0), the probability of choosing the exploration and exploitation phase, default=0.18
    """
    c1: float
    c2: float
    pp: float | None = 0.18

    @field_validator("c1")
    def correct_c1(cls, v):
        if not 0 <= v <= 0.18:
            raise ValueError(f"\"c1\" must be in (-100., 100.). Got {v}")
        return v

    @field_validator("c2")
    def correct_c2(cls, v):
        if not 0.19 <= v <= 1:
            raise ValueError(f"\"c2\" must be in (-100., 100.). Got {v}")
        return v

    @field_validator("pp")
    def correct_pp(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"pp\" must be in (0.0, 1.0). Got {v}")
        return v
