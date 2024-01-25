from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Patient(Agent):
    status: int | None = 0
    age: int | None = 0


class CoronavirusHerdImmunityOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Particle Swarm Optimization algorithm.
        C0 (float): (0, 1.0), initial concentration of the virus.\n
        brr (float): [0.05, 0.2], basic reproduction rate.\n
        max_age (int): [5, 20], maximum infected cases age.
    """
    C0: float
    brr: float
    max_age: int

    @field_validator("C0")
    def check_C0(cls, v):
        if not 0 < v < 1.0:
            raise ValueError(f"\"C0\" must be in range (0, 1.0). Got {v}")
        return v

    @field_validator("brr")
    def check_brr(cls, v):
        if not 0.05 <= v <= 0.2:
            raise ValueError(f"\"brr\" must be in range [0.05, 0.2]. Got {v}")
        return v

    @field_validator("max_age")
    def check_max_age(cls, v):
        if not 5 <= v <= 20:
            raise ValueError(f"\"max_age\" must be in range [5, 20]. Got {v}")
        return v
