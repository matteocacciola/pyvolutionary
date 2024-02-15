from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Molecule(Agent):
    pass


class FicksLawOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Fick's Law Optimization algorithm.
        + C1 (float): (-100., 100.), factor C1.\n
        + C2 (float): (-100., 100.), factor C2.\n
        + C3 (float): (-100., 100.), factor C3.\n
        + C4 (float): (-100., 100.), factor C4.\n
        + C5 (float): (-100., 100.), factor C5.\n
        + DD (float): (-100., 100.), factor D in the paper.
    """
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    DD: float

    @field_validator("C1")
    def correct_C1(cls, v):
        if not -100 < v < 100:
            raise ValueError(f"\"C1\" must be a float in (-100., 100.). Got {v}")
        return v

    @field_validator("C2")
    def correct_C2(cls, v):
        if not -100 < v < 100:
            raise ValueError(f"\"C2\" must be a float in (-100., 100.). Got {v}")
        return v

    @field_validator("C3")
    def correct_C3(cls, v):
        if not -100 < v < 100:
            raise ValueError(f"\"C3\" must be a float in (-100., 100.). Got {v}")
        return v

    @field_validator("C4")
    def correct_C4(cls, v):
        if not -100 < v < 100:
            raise ValueError(f"\"C4\" must be a float in (-100., 100.). Got {v}")
        return v

    @field_validator("C5")
    def correct_C5(cls, v):
        if not -100 < v < 100:
            raise ValueError(f"\"C5\" must be a float in (-100., 100.). Got {v}")
        return v

    @field_validator("DD")
    def correct_DD(cls, v):
        if not -100 < v < 100:
            raise ValueError(f"\"DD\" must be a float in (-100., 100.). Got {v}")
        return v
