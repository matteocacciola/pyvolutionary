from pydantic import conlist, field_validator

from ..models import Agent, BaseOptimizationConfig


class Cat(Agent):
    velocity: list[float]
    flag: bool = False


class CatSwarmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Particle Swarm Optimization algorithm.
        mixture_ratio (float): (0, 1.0), joining seeking mode with tracing mode.\n
        smp (int): [2, 10000], seeking memory pool, 10 clones (larger is better but time-consuming).\n
        spc (bool): self-position considering.\n
        cdc (float): (0, 1.0), counts of dimension to change  (larger is more diversity but slow convergence).\n
        srd (float): (0, 1.0), seeking range of the selected dimension (smaller is better but slow convergence).\n
        c1 (float): (0, 3), cognitive parameter.\n
        w (list[float]): [0.1, 0.5], [0.5, 2.0], weights.\n
        selected_strategy (int): 0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)
    """
    c1: float
    mixture_ratio: float
    smp: int
    spc: bool = True
    cdc: float
    srd: float
    w: conlist(float, min_length=2, max_length=2)
    selected_strategy: int = 0

    @field_validator("mixture_ratio")
    def correct_mixture_ratio(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"mixture_ratio\" must be a float in (0, 1.0). Got {v}")
        return v

    @field_validator("smp")
    def correct_smp(cls, v):
        if not 2 <= v <= 10000:
            raise ValueError(f"\"smp\" must be an int in [2, 10000]. Got {v}")
        return v

    @field_validator("cdc")
    def correct_cdc(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"cdc\" must be a float in (0, 1.0). Got {v}")
        return v

    @field_validator("srd")
    def correct_srd(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"\"srd\" must be a float in (0, 1.0). Got {v}")
        return v

    @field_validator("c1")
    def correct_c1(cls, v):
        if not 0 < v < 3:
            raise ValueError(f"\"c1\" must be a float in (0, 5.0). Got {v}")
        return v

    @field_validator("w")
    def correct_weights(cls, v):
        w_min, w_max = v
        if not w_min <= w_max:
            raise ValueError(f"\"w[0]\" must be less or equal than \"w[1]\". Got {w_min} and {w_max}")
        if not 0.1 <= w_min <= 0.5:
            raise ValueError(f"\"w[0]\" must be a float in [0.1, 0.5]. Got {w_min}")
        if not 0.5 <= w_max <= 2:
            raise ValueError(f"\"w[1]\" must be a float in [0.5, 2.0]. Got {w_max}")
        return v
