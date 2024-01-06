from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Cell(Agent):
    local_cost: float
    local_best: list[float]
    nutrients: float = 0.0


class BacterialForagingOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Adaptive Bacterial Foraging Optimization Algorithm.
        C_s (float): (0, 2.0), step size start.\n
        C_e (float): (0, 1.0), step size end.\n
        Ped (float): (0, 1.0), probability eliminate.\n
        Ns (int): [2, 100], swim_length.\n
        N_adapt (int): [0, 4], dead threshold value.\n
        N_split (int): [5, 50], split threshold value.
    """
    C_s: float
    C_e: float
    Ped: float
    Ns: int
    N_adapt: int
    N_split: int

    @field_validator("C_s")
    def check_C_s(cls, v):
        if not (0 < v < 2.0):
            raise ValueError("C_s must be in (0, 2.0)")
        return v

    @field_validator("C_e")
    def check_C_e(cls, v):
        if not (0 < v < 1.0):
            raise ValueError("C_e must be in (0, 1.0)")
        return v

    @field_validator("Ped")
    def check_Ped(cls, v):
        if not (0 < v < 1.0):
            raise ValueError("Ped must be in (0, 1.0)")
        return v

    @field_validator("Ns")
    def check_Ns(cls, v):
        if not (2 <= v <= 100):
            raise ValueError("Ns must be in [2, 100]")
        return v

    @field_validator("N_adapt")
    def check_N_adapt(cls, v):
        if not (0 <= v <= 4):
            raise ValueError("N_adapt must be in [0, 4]")
        return v

    @field_validator("N_split")
    def check_N_split(cls, v):
        if not (5 <= v <= 50):
            raise ValueError("N_split must be in [5, 50]")
        return v
