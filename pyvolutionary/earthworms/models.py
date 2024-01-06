from pydantic import field_validator, model_validator

from ..models import Agent, BaseOptimizationConfig


class Earthworm(Agent):
    pass


class EarthwormsOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Earthworms Optimization algorithm.
        prob_mutate (float): [0, 1), probability of mutation of an earthworm.\n
        prob_crossover (float): [0, 1), probability of crossover of an earthworm.\n
        keep (int): [2, int(population_size/2)], number of earthworms to keep.\n
        alpha (float): [0, 1), similarity factor.\n
        beta (float): [0, 1), the initial proportional factor.\n
        gamma (float): [0, 1), a sort cooling factor for beta.
    """
    prob_mutate: float
    prob_crossover: float
    keep: int
    alpha: float
    beta: float
    gamma: float

    @model_validator(mode="after")
    def validate_keep(self) -> "EarthwormsOptimizationConfig":
        if not 2 <= self.keep <= int(self.population_size / 2):
            raise ValueError(f"keep must be a positive integer greater than 2 and lower than population_size / 2. "
                             f"Got {self.keep}")
        return self

    @field_validator("prob_mutate")
    def correct_prob_mutate(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"prob_mutate\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("prob_crossover")
    def correct_prob_crossover(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"prob_crossover\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"alpha\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("beta")
    def correct_beta(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"beta\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("gamma")
    def correct_gamma(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"gamma\" must be a positive float lower than 1. Got {v}")
        return v
