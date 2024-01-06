from pydantic import model_validator, field_validator

from ..models import Agent, BaseOptimizationConfig


class Tree(Agent):
    age: int = 0


class ForestOptimizationAlgorithmConfig(BaseOptimizationConfig):
    """
    Configuration class of the Forest Optimization Algorithm.
        lifetime (int): the lifetime of a tree.\n
        area_limit (int): the area limit of a tree.\n
        local_seeding_changes (int): the number of local seeding changes.\n
        global_seeding_changes (int): the number of global seeding changes.\n
        transfer_rate (float): (0, 1), the transfer rate.
    """
    lifetime: int = 10
    area_limit: int = 10
    local_seeding_changes: int = 3
    global_seeding_changes: int = 3
    transfer_rate: float = 0.5

    @model_validator(mode="after")
    def _check_area_limit(self) -> "ForestOptimizationAlgorithmConfig":
        if not 0 < self.area_limit < self.population_size:
            raise ValueError('area_limit must be a positive integer, lower than the population size')
        return self

    @field_validator('lifetime')
    def _check_lifetime(cls, v):
        if v < 0:
            raise ValueError('lifetime must be a positive integer')
        return v

    @field_validator('local_seeding_changes')
    def _check_local_seeding_changes(cls, v):
        if v < 0:
            raise ValueError('local_seeding_changes must be a positive integer')
        return v

    @field_validator('global_seeding_changes')
    def _check_global_seeding_changes(cls, v):
        if v < 0:
            raise ValueError('global_seeding_changes must be a positive integer')
        return v

    @field_validator('transfer_rate')
    def _check_transfer_rate(cls, v):
        if not 0 < v < 1:
            raise ValueError('transfer_rate must be a float between 0 and 1')
        return v
