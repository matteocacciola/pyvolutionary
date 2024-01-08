from pydantic import conlist, field_validator

from ..models import Agent, BaseOptimizationConfig


class Particle(Agent):
    velocity: list[float]


class ParticleSwarmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Particle Swarm Optimization algorithm.
        c1 (float): (0, 5), cognitive parameter.\n
        c2 (float): (0, 5), social parameter.\n
        w (list[float]): (0, 0.5), [0.5, 2.0], weights.
    """
    c1: float
    c2: float
    w: conlist(float, min_length=2, max_length=2)

    @field_validator("c1")
    def correct_c1(cls, v):
        if not 0 < v < 5:
            raise ValueError(f"\"c1\" must be a float in (0, 5.0). Got {v}")
        return v

    @field_validator("c2")
    def correct_c2(cls, v):
        if not 0 < v < 5:
            raise ValueError(f"\"c2\" must be a float in (0, 5.0). Got {v}")
        return v

    @field_validator("w")
    def correct_weights(cls, v):
        w_min, w_max = v
        if not w_min < w_max:
            raise ValueError(f"\"w[0]\" must be less than \"w[1]\". Got {w_min} and {w_max}")
        if not 0 < w_min < 0.5:
            raise ValueError(f"\"w[0]\" must be a float in (0, 0.5). Got {w_min}")
        if not 0.5 <= w_max <= 2:
            raise ValueError(f"\"w[1]\" must be a float in [0.5, 2.0]. Got {w_max}")
        return v
