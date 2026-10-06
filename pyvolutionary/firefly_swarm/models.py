from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Firefly(Agent):
    pass


class FireflySwarmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Firefly Swarm Optimization algorithm.
        alpha (float): [0, 1), the mutation coefficient, i.e. the weight of the random step of the fireflies.\n
        beta_min (float): [0, 3], the base value of the attractiveness between the fireflies (beta_0 in the paper).\n
        gamma (float): [0, 1), the light absorption coefficient.\n
        alpha_damp (float): (0, 1], the damping rate of the mutation coefficient at each cycle, default = 0.99.\n
        delta (float): (0, 1], the step size of the random step of the fireflies, default = 0.05.\n
        exponent (int): [2, 4], the exponent of the distance in the attractiveness (m in the paper), default = 2.
    """
    alpha: float
    beta_min: float
    gamma: float
    alpha_damp: float = 0.99
    delta: float = 0.05
    exponent: int = 2

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"alpha\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("beta_min")
    def correct_beta_min(cls, v):
        if not 0 <= v <= 3:
            raise ValueError(f"\"beta_min\" must be a positive float not greater than 3. Got {v}")
        return v

    @field_validator("gamma")
    def correct_gamma(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"gamma\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("alpha_damp")
    def correct_alpha_damp(cls, v):
        if not 0 < v <= 1:
            raise ValueError(f"\"alpha_damp\" must be a float in (0, 1]. Got {v}")
        return v

    @field_validator("delta")
    def correct_delta(cls, v):
        if not 0 < v <= 1:
            raise ValueError(f"\"delta\" must be a float in (0, 1]. Got {v}")
        return v

    @field_validator("exponent")
    def correct_exponent(cls, v):
        if not 2 <= v <= 4:
            raise ValueError(f"\"exponent\" must be an integer in [2, 4]. Got {v}")
        return v
