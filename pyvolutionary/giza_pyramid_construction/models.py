from pydantic import conlist, field_validator

from ..models import Agent, BaseOptimizationConfig


class Worker(Agent):
    pass


class GizaPyramidConstructionOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Giza Pyramid Construction Optimization algorithm.
        gravity (float): gravity acceleration, default 9.8.\n
        theta (float): (10, 45), angle of the ramp.\n
        friction (list[float]): [1, 100), [1, 100), friction coefficients.\n
        prob_substitution (float): (0, 1), probability of substitution.
    """
    gravity: float = 9.8
    theta: float
    friction: conlist(float, min_length=2, max_length=2)
    prob_substitution: float

    @field_validator("theta")
    def correct_theta(cls, v):
        if not 10 < v < 45:
            raise ValueError(f"\"theta\" must be a float in (10, 45). Got {v}")
        return v

    @field_validator("friction")
    def correct_friction(cls, v):
        mu1, mu2 = v
        if not mu1 < mu2:
            raise ValueError(f"\"friction[0]\" must be less than \"friction[1]\". Got {mu1} and {mu2}")
        if not 1 <= mu1 < 100:
            raise ValueError(f"\"friction[0]\" must be a float in [1, 100). Got {mu1}")
        if not 1 <= mu2 < 100:
            raise ValueError(f"\"friction[1]\" must be a float in [1, 100). Got {mu2}")
        return v

    @field_validator("prob_substitution")
    def correct_prob_substitution(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"prob_substitution\" must be a float in (0, 1). Got {v}")
        return v
