from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class DwarfMongoose(Agent):
    C: float | None = 0.


class DwarfMongooseOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Dwarf Mongoose Optimization algorithm.
        n_baby_sitter (int): [2, 10], number of babysitters.\n
        peep (float): [1, 10], intensity of peeps.\n
    """
    n_baby_sitter: int
    peep: float

    @field_validator("n_baby_sitter")
    def check_valid_n_baby_sitter(cls, v):
        if not 2 <= v <= 10:
            raise ValueError(f"n_baby_sitter must be an integer in [2, 10]. Got {v}.")
        return v

    @field_validator("peep")
    def check_valid_peep(cls, v):
        if not 1 <= v <= 10:
            raise ValueError(f"peep must be a float in [1, 10]. Got {v}.")
        return v
