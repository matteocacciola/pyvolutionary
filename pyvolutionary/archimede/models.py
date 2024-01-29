from typing import Any
from pydantic import conlist, field_validator

from ..models import Agent, BaseOptimizationConfig


class Object(Agent):
    density: list[Any]
    volume: list[Any]
    acceleration: list[Any]


class ArchimedeOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Archimede Optimization algorithm.
        c1 (int): [1, 3], factor.\n
        c2 (int): [2, 6], factor.\n
        c3 (int): [1, 3], factor.\n
        c4 (float): (0, 1.0), factor.\n
        acc (list[float]): (0, 0.3), [0.3, 1.0], min and max acceleration.\n
    """
    c1: int
    c2: int
    c3: int
    c4: float
    acc: conlist(float, min_length=2, max_length=2)

    @field_validator("c1")
    def correct_c1(cls, v):
        if not 1 <= v <= 3:
            raise ValueError(f"\"c1\" must be an integer in [1, 3]. Got {v}")
        return v

    @field_validator("c2")
    def correct_c2(cls, v):
        if not 2 <= v <= 6:
            raise ValueError(f"\"c2\" must be an integer in [2, 6]. Got {v}")
        return v

    @field_validator("c3")
    def correct_c3(cls, v):
        if not 1 <= v <= 3:
            raise ValueError(f"\"c3\" must be an integer in [1, 3]. Got {v}")
        return v

    @field_validator("c4")
    def correct_c4(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"c4\" must be a float in (0, 1.0). Got {v}")
        return v

    @field_validator("acc")
    def correct_acc(cls, v):
        acc_min, acc_max = v
        if not acc_min < acc_max:
            raise ValueError(f"\"acc[0]\" must be less than \"acc[1]\". Got {acc_min} and {acc_max}")
        if not 0 < acc_min < 0.3:
            raise ValueError(f"\"acc[0]\" must be a float in (0, 0.3). Got {acc_min}")
        if not 0.3 <= acc_max <= 1:
            raise ValueError(f"\"acc[1]\" must be a float in [0.3, 1.0]. Got {acc_max}")
        return v
