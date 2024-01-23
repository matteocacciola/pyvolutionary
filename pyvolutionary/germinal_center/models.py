from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class GerminalCenter(Agent):
    cell_counter: float | None = 1.0
    life_signal: float | None = 70.0  # 70% to duplicate, and 30% to die

    def __init__(self, **kwargs):
        kwargs["cell_counter"] = kwargs.get("cell_counter") or 1.0
        kwargs["life_signal"] = kwargs.get("life_signal") or 70.0
        super().__init__(**kwargs)


class GerminalCenterOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Germinal Center Optimization algorithm.
        cr (float): (0, 1.0), crossover rate (same as DE algorithm).\n
        wf (float): (0, 3.0), weighting factor (f in the paper, same as DE algorithm).
    """
    cr: float
    wf: float

    @field_validator("cr")
    def correct_cr(cls, v):
        if not 0 < v < 1.0:
            raise ValueError(f"\"cr\" must be in (0, 1.0). Got {v}")
        return v

    @field_validator("wf")
    def correct_wf(cls, v):
        if not 0 < v < 3.0:
            raise ValueError(f"\"wf\" must be in (0, 3.0). Got {v}")
        return v
