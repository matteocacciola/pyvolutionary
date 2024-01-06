from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Empire(Agent):
    pass


class ImperialistCompetitiveOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Imperialist Competitive Optimization algorithm.
        assimilation_rate (float): [0, 1), the rate of assimilation of the colonies by the imperialist.\n
        revolution_rate (float): [0, 1), the rate of revolution of the colonies.\n
        alpha_rate (float): [0, 1), the rate of alpha.\n
        revolution_probability (float): [0, 1), the probability of revolution.\n
        number_of_countries (int): the number of countries per empire.\n
    """
    assimilation_rate: float
    revolution_rate: float
    alpha_rate: float
    revolution_probability: float
    number_of_countries: int

    @field_validator("assimilation_rate")
    def correct_assimilation_rate(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"assimilation_rate\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("revolution_rate")
    def correct_revolution_rate(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"revolution_rate\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("alpha_rate")
    def correct_alpha_rate(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"alpha_rate\" must be a positive float lower than 1. Got {v}")
        return v

    @field_validator("revolution_probability")
    def correct_revolution_probability(cls, v):
        if not 0 <= v < 1:
            raise ValueError(f"\"revolution_probability\" must be a positive float lower than 1. Got {v}")
        return v
