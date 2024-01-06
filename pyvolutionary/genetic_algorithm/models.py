from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Gene(Agent):
    pass


class GeneticAlgorithmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Genetic Algorithm.
        px_over (float): [0, 1), probability of crossover between two genes.
    """
    px_over: float

    @field_validator("px_over")
    def correct_px_over(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"px_over\" must be a positive float lower than 1. Got {v}")
        return v
