from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Krill(Agent):
    induced_speed: list[float]
    foraging_speed: list[float]


class KrillHerdOptimizationConfig(BaseOptimizationConfig):
    """
    Krill Herd Optimization Configuration
        n_max (float): maximum induced speed.\n
        foraging_speed (float): Foraging speed.\n
        diffusion_speed (float): maximum diffusion speed.\n
        c_t (float): [0, 2] constant.\n
        w_neighbour (float): [0, 1] inertia weights of the motion induced from neighbors.\n
        w_foraging (float): [0, 1] inertia weights of the motion induced from foraging.\n
        max_neighbours (int): maximum neighbors for neighbors effect.\n
        crossover_rate (float): (0, 1) crossover probability.\n
        mutation_rate (float): (0, 1) mutation probability.\n
    """
    n_max: float
    foraging_speed: float
    diffusion_speed: float
    c_t: float
    w_neighbour: float
    w_foraging: float
    max_neighbours: int
    crossover_rate: float
    mutation_rate: float

    @field_validator("c_t")
    def correct_c_t(cls, v):
        if not 0 <= v <= 2:
            raise ValueError(f"\"c_t\" must be a float in [0, 2]. Got {v}")
        return v

    @field_validator("w_neighbour")
    def correct_w_neighbour(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"\"w_neighbour\" must be a float in [0, 1]. Got {v}")
        return v

    @field_validator("w_foraging")
    def correct_w_foraging(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"\"w_foraging\" must be a float in [0, 1]. Got {v}")
        return v

    @field_validator("crossover_rate")
    def correct_crossover_rate(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"crossover_rate\" must be a float in (0, 1). Got {v}")
        return v

    @field_validator("mutation_rate")
    def correct_mutation_rate(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"mutation_rate\" must be a float in (0, 1). Got {v}")
        return v
