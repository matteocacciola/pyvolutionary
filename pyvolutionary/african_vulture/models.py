from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class AfricanVulture(Agent):
    pass


class AfricanVultureOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the African Vulture Optimization algorithm.
        p (list[float]): (0, 1), (0, 1), (0, 1), list of 3 floats in [0, 1] that represent the probabilities of each of
        the 3 status transition (see paper).\n
        alpha (float): (0, 1), the alpha parameter of the algorithm, i.e. probability of 1st best (see paper).\n
        gamma (float): (0, 5), the gamma parameter of the algorithm (see paper).
    """
    p: list[float]
    alpha: float
    gamma: float

    @field_validator("p")
    def correct_p(cls, v):
        if len(v) != 3:
            raise ValueError(f"\"p\" must be a list of 3 floats. Got {v}")
        # check if any of v is not in [0, 1]
        if any([not (0 < x < 1) for x in v]):
            raise ValueError(f"\"p\" must be a list of 3 floats in [0, 1]. Got {v}")
        return v

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not (0 < v < 1):
            raise ValueError(f"\"alpha\" must be a float in (0, 1). Got {v}")
        return v

    @field_validator("gamma")
    def correct_gamma(cls, v):
        if not (0 < v < 5):
            raise ValueError(f"\"gamma\" must be a float in (0, 5). Got {v}")
        return v
