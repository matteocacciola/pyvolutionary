from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Coral(Agent):
    pass


class CoralReefOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Coral Reef algorithm.
        po (float): [0.2, 0.5], the rate between free/occupied at the beginning.\n
        Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate.\n
        Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the
        reef.\n
        Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation.\n
        Pd (float): [0.1, 0.7], Probability of depredation.\n
        GCR (float): [0.05, 0.2], probability for mutation process.\n
        gamma (list[float]): [0.01, 0.1], [0.1, 0.5], min and max factors for mutation process.\n
        n_trials (int): [2, 10], number of attempts for a larvae to set in the reef.
    """
    po: float
    Fb: float
    Fa: float
    Fd: float
    Pd: float
    GCR: float
    gamma: list[float]
    n_trials: int

    @field_validator('po')
    def check_po(cls, v):
        if not 0.2 <= v <= 0.5:
            raise ValueError('po must be in range [0.2, 0.5]')
        return v

    @field_validator('Fb')
    def check_Fb(cls, v):
        if not 0.6 <= v <= 0.9:
            raise ValueError('Fb must be in range [0.6, 0.9]')
        return v

    @field_validator('Fa')
    def check_Fa(cls, v):
        if not 0.05 <= v <= 0.3:
            raise ValueError('Fa must be in range [0.05, 0.3]')
        return v

    @field_validator('Fd')
    def check_Fd(cls, v):
        if not 0.05 <= v <= 0.5:
            raise ValueError('Fd must be in range [0.05, 0.5]')
        return v

    @field_validator('Pd')
    def check_Pd(cls, v):
        if not 0.1 <= v <= 0.7:
            raise ValueError('Pd must be in range [0.1, 0.7]')
        return v

    @field_validator('GCR')
    def check_GCR(cls, v):
        if not 0.05 <= v <= 0.2:
            raise ValueError('GCR must be in range [0.05, 0.2]')
        return v

    @field_validator('gamma')
    def check_gamma(cls, v):
        gamma_min, gamma_max = v
        if not 0.01 <= gamma_min <= 0.1:
            raise ValueError('gamma_min must be in range [0.01, 0.1]')
        if not 0.1 <= gamma_max <= 0.5:
            raise ValueError('gamma_max must be in range [0.1, 0.5]')
        return v

    @field_validator('n_trials')
    def check_n_trials(cls, v):
        if not 2 <= v <= 10:
            raise ValueError('n_trials must be in range [2, 10]')
        return v
