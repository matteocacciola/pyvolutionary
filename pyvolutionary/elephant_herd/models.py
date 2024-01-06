from pydantic import field_validator, model_validator

from ..models import Agent, BaseOptimizationConfig


class Elephant(Agent):
    pass


class ElephantHerdOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Earthworms Optimization algorithm.
        alpha (float): (0, 3), a factor that determines the influence of the best in each clan.\n
        beta (float): (0, 1), a factor that determines the influence of the x_center.\n
        n_clans (int): [2, int(population_size/5)], the number of clans.
    """
    alpha: float
    beta: float
    n_clans: int

    @model_validator(mode="after")
    def validate_n_clans(self) -> "ElephantHerdOptimizationConfig":
        if not 2 <= self.n_clans <= int(self.population_size / 5):
            raise ValueError(f"n_clans must be a positive integer greater than 2 and lower than population_size / 5. "
                             f"Got {self.n_clans}")
        return self

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not 0 < v < 3:
            raise ValueError(f"\"alpha\" must be a positive float lower than 3. Got {v}")
        return v

    @field_validator("beta")
    def correct_beta(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"beta\" must be a positive float lower than 1. Got {v}")
        return v
