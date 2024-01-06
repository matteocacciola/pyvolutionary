from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Wildebeest(Agent):
    pass


class WildebeestHerdOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Wildebeest Herd Optimization algorithm.
        n_explore_step (int): [2, 10] -> better [2, 4], number of exploration step\n
        n_exploit_step (int): [2, 10] -> better [2, 4], number of exploitation step\n
        eta (float): (0, 1.0) -> better [0.05, 0.5], learning rate\n
        phi (float): (0, 1.0) -> better [0.7, 0.95], the probability of wildebeest move to another position based on
        herd instinct\n
        local_alpha (float): (0, 3.0) -> better [0.5, 0.9], control local movement\n
        local_beta (float): (0, 3.0) -> better [0.1, 0.5], control local movement\n
        global_alpha (float): (0, 3.0) -> better [0.1, 0.5], control global movement\n
        global_beta (float): (0, 3.0), control global movement\n
        delta_w (float): (0.5, 5.0) -> better [1.0, 2.0], dist to worst\n
        delta_c (float): (0.5, 5.0) -> better [1.0, 2.0], dist to best
    """
    n_explore_step: int
    n_exploit_step: int
    eta: float
    phi: float
    local_alpha: float
    local_beta: float
    global_alpha: float
    global_beta: float
    delta_w: float
    delta_c: float

    @field_validator("n_explore_step")
    def correct_n_explore_step(cls, v):
        if not 2 <= v <= 10:
            raise ValueError(f"\"n_explore_step\" must be an int in [2, 10]. Got {v}")
        return v

    @field_validator("n_exploit_step")
    def correct_n_exploit_step(cls, v):
        if not 2 <= v <= 10:
            raise ValueError(f"\"n_exploit_step\" must be an int in [2, 10]. Got {v}")
        return v

    @field_validator("eta")
    def correct_eta(cls, v):
        if not 0 < v <= 1.0:
            raise ValueError(f"\"eta\" must be a float in (0, 1.0]. Got {v}")
        return v

    @field_validator("phi")
    def correct_phi(cls, v):
        if not 0 < v <= 1.0:
            raise ValueError(f"\"phi\" must be a float in (0, 1.0]. Got {v}")
        return v

    @field_validator("local_alpha")
    def correct_local_alpha(cls, v):
        if not 0 < v <= 3.0:
            raise ValueError(f"\"local_alpha\" must be a float in (0, 3.0]. Got {v}")
        return v

    @field_validator("local_beta")
    def correct_local_beta(cls, v):
        if not 0 < v <= 3.0:
            raise ValueError(f"\"local_beta\" must be a float in (0, 3.0]. Got {v}")
        return v

    @field_validator("global_alpha")
    def correct_global_alpha(cls, v):
        if not 0 < v <= 3.0:
            raise ValueError(f"\"global_alpha\" must be a float in (0, 3.0]. Got {v}")
        return v

    @field_validator("global_beta")
    def correct_global_beta(cls, v):
        if not 0 < v <= 3.0:
            raise ValueError(f"\"global_beta\" must be a float in (0, 3.0]. Got {v}")
        return v

    @field_validator("delta_w")
    def correct_delta_w(cls, v):
        if not 0.5 < v <= 5.0:
            raise ValueError(f"\"delta_w\" must be a float in (0.5, 5.0]. Got {v}")
        return v

    @field_validator("delta_c")
    def correct_delta_c(cls, v):
        if not 0.5 < v <= 5.0:
            raise ValueError(f"\"delta_c\" must be a float in (0.5, 5.0]. Got {v}")
        return v
