from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Pollinator(Agent):
    pass


class FlowerPollinationAlgorithmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Flower Pollination Algorithm.
        p_s (float): [0.5, 0.95], switch probability.\n
        levy_multiplier (float): [0.0001, 1000], multiplier factor of Levy-flight trajectory, depends on the problem.
    """
    p_s: float
    levy_multiplier: float

    @field_validator("p_s")
    def correct_p_s(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError(f"\"p_s\" must be a float between 0.5 and 0.95. Got {v}")
        return v

    @field_validator("levy_multiplier")
    def correct_levy_multiplier(cls, v):
        if not 0.0001 <= v <= 1000:
            raise ValueError(f"\"levy_multiplier\" must be a float between 0.0001 and 1000. Got {v}")
        return v
