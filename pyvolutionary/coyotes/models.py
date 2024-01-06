from pydantic import model_validator

from ..models import Agent, BaseOptimizationConfig


class Coyote(Agent):
    age: int


class CoyotesOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Coyotes Optimization algorithm.
        num_coyotes (int): [2, int(population_size/2)], the number of coyotes per group.
    """
    num_coyotes: int

    @model_validator(mode="after")
    def validate_keep(self) -> "CoyotesOptimizationConfig":
        if not 2 <= self.num_coyotes <= int(self.population_size / 2):
            raise ValueError(f"num_coyotes must be greater than 2 and lower than population_size / 2. "
                             f"Got {self.num_coyotes}")
        return self
