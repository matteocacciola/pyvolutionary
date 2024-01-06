from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Firefly(Agent):
    pass


class FireflySwarmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Firefly Swarm Optimization algorithm.
        alpha (float): [0, 1), the attractiveness constant of the fireflies.\n
        beta_min (float): [0, 1), the minimum value of the attractiveness constant of the fireflies.\n
        gamma (float): [0, 1), the light absorption coefficient.
    """
    alpha: float
    beta_min: float
    gamma: float

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"alpha\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("beta_min")
    def correct_beta_min(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"beta_min\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("gamma")
    def correct_gamma(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"gamma\" must be a positive float lower than 1. Got {v}")
        return v
