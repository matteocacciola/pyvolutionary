from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class AirParcel(Agent):
    pass


class WindDrivenOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Wind Driven Optimization algorithm.
        RT (int): [2, 4], RT coefficient.\n
        g_c (float): [0.1, 0.5], gravitational constant.\n
        alp (float): [0.3, 0.8], constants in the update equation.\n
        c_e (float): [0.1, 0.9], coriolis effect.\n
        max_v (float): [0.1, 0.9], maximum allowed speed.
    """
    RT: int
    g_c: float
    alp: float
    c_e: float
    max_v: float

    @field_validator("RT")
    def correct_RT(cls, v):
        if not 2 <= v <= 4:
            raise ValueError(f"\"RT\" must be an int in [2, 4]. Got {v}")
        return v

    @field_validator("g_c")
    def correct_g_c(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError(f"\"g_c\" must be a float in [0.1, 0.5]. Got {v}")
        return v

    @field_validator("alp")
    def correct_alp(cls, v):
        if not 0.3 <= v <= 0.8:
            raise ValueError(f"\"alp\" must be a float in [0.3, 0.8]. Got {v}")
        return v

    @field_validator("c_e")
    def correct_c_e(cls, v):
        if not 0.1 <= v <= 0.9:
            raise ValueError(f"\"c_e\" must be a float in [0.1, 0.9]. Got {v}")
        return v

    @field_validator("max_v")
    def correct_max_v(cls, v):
        if not 0.1 <= v <= 0.9:
            raise ValueError(f"\"max_v\" must be a float in [0.1, 0.9]. Got {v}")
        return v
