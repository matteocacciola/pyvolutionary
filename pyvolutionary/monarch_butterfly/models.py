from pydantic import model_validator

from ..models import Agent, BaseOptimizationConfig


class MonarchButterfly(Agent):
    pass


class MonarchButterflyOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Monarch Butterfly Optimization algorithm.
        partition (float): the value of the partition.\n
        period (float): the period as per the description in the original paper.\n
        keep (int): elitism parameter: how many of the best habitats to keep from one generation to the next
    """
    partition: float
    period: float
    keep: int = 2

    @model_validator(mode="after")
    def validate_keep(self) -> "MonarchButterflyOptimizationConfig":
        if not 2 <= self.keep < self.population_size / 2:
            raise ValueError(f"keep must be between 2 and population_size / 2, got {self.keep}")
        return self
