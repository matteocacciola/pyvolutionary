import numpy as np
from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Camel(Agent):
    endurance: float
    supply: float
    temperature: float | None = None
    steps: int = 1


class CamelCaravanOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Camel Caravan algorithm.
        burden_factor (float): [0, 1], burden factor.\n
        death_rate (float): [0, 1], dying rate.\n
        visibility (float): view range of camel.\n
        supply (float): (0, Inf), initial supply.\n
        endurance (float): (0, Inf), initial endurance.\n
        temperatures (list[float]): minimum and maximum temperatures.\n
    """
    burden_factor: float
    death_rate: float
    visibility: float
    supply: float
    endurance: float
    temperatures: list[float]

    @field_validator('burden_factor')
    def check_burden_factor(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('burden_factor must be in range [0, 1]')
        return v

    @field_validator('death_rate')
    def check_death_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('death_rate must be in range [0, 1]')
        return v

    @field_validator('supply')
    def check_supply(cls, v):
        if not 0 < v:
            raise ValueError('supply must be greater than 0')
        return v

    @field_validator('endurance')
    def check_endurance(cls, v):
        if not 0 < v:
            raise ValueError('endurance must be greater than 0')
        return v

    @field_validator('temperatures')
    def check_temperatures(cls, v):
        if not len(v) == 2:
            raise ValueError('temperatures must be a list of length 2')
        if not v[0] < v[1]:
            raise ValueError('temperatures[0] must be less than temperatures[1]')
        return v