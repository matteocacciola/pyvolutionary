from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class SpottedHyena(Agent):
    pass


class SpottedHyenaOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Spotted Hyena Optimization algorithm.
        h_factor (float): (0.5, 10.0), coefficient linearly decreased from 5 to 0.\n
        n_trials (int): (1, Inf(, number of trials for each agent.\n
    """
    h_factor: float
    n_trials: int

    @field_validator("h_factor")
    def correct_h_factor(cls, v):
        if not 0.5 < v < 10:
            raise ValueError(f"\"h_factor\" must be a float in (0.5, 10.0). Got {v}")
        return v

    @field_validator("n_trials")
    def correct_n_trials(cls, v):
        if not 1 < v:
            raise ValueError(f"\"n_trials\" must be an int greater than 1. Got {v}")
        return v
