from pydantic import field_validator, ConfigDict

from .classes import QTable
from ..models import Agent, BaseOptimizationConfig


class Candidate(Agent):
    pass


class QleCandidate(Agent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    qtable: QTable


class SineCosineAlgorithmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Sine Cosine Algorithm Optimization.
    """
    pass


class QleSineCosineAlgorithmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for QLE Sine Cosine Algorithm Optimization.
        alpha (float): [0.1, 1.0], it is the learning rate in Q-learning.\n
        gama (float): [0.1-1.0]: the discount factor.
    """
    alpha: float
    gama: float

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError(f"\"alpha\" must be in range [0.1, 1.0]. Got {v}")
        return v

    @field_validator("gama")
    def correct_gamma(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError(f"\"gama\" must be in range [0.1, 1.0]. Got {v}")
        return v
