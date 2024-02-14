from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Gas(Agent):
    pass


class HenryGasSolubilityOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Henry Gas Solubility Optimization algorithm.
        n_clusters (int): [2, 10], number of clusters.
    """
    n_clusters: int

    @field_validator("n_clusters")
    def correct_n_clusters(cls, v):
        if not 2 <= v <= 10:
            raise ValueError(f"\"n_clusters\" must be a positive integer in [2, 10]. Got {v}")
        return v
