from pydantic import field_validator, model_validator

from ..models import Agent, BaseOptimizationConfig


class Population(Agent):
    pass


class BiogeographyBasedOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Biogeography Based Optimization algorithm.
        p_m (float): (0, 1), Mutation probability.\n
        n_elites (int): (2, pop_size/2), Number of elites will be keep for next generation
    """
    p_m: float
    n_elites: int

    @field_validator("p_m")
    def check_p_m(cls, v):
        if not 0 < v < 1:
            raise ValueError("Mutation probability must be in (0, 1)")
        return v

    @model_validator(mode="after")
    def validate_n_elites(self) -> "BiogeographyBasedOptimizationConfig":
        if not 2 < self.n_elites < self.population_size / 2:
            raise ValueError("Number of elites must be in (2, pop_size/2)")
        return self
