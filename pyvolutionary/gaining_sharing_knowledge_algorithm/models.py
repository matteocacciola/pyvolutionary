from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Knowledge(Agent):
    pass


class GainingSharingKnowledgeOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Gaining Sharing Knowledge-based Optimization algorithm.
        p (float): [0.1, 0.5], percent of the best.\n
        kf (float): [0.3, 0.8], knowledge factor that controls the total amount of gained and shared knowledge added
            from others to the current individual during generations.\n
        kr (float): [0.5, 0.95], knowledge ratio, default = 0.9.\n
        kg (int): [3, 20], number of generations effect to D-dimension.
    """
    p: float
    kf: float
    kr: float
    kg: float

    @field_validator("p")
    def correct_p(cls, v):
        if not 0 <= v <= 0.5:
            raise ValueError(f"\"p\" must be a float in [0.1, 0.5]. Got {v}")
        return v

    @field_validator("kf")
    def correct_kf(cls, v):
        if not 0.3 <= v <= 0.8:
            raise ValueError(f"\"kf\" must be a float in [0.3, 0.8]. Got {v}")
        return v

    @field_validator("kr")
    def correct_kr(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError(f"\"kf\" must be a float in [0.5, 0.95]. Got {v}")
        return v

    @field_validator("kg")
    def correct_kg(cls, v):
        if not 3 <= v <= 20:
            raise ValueError(f"\"kf\" must be a int in [3, 20]. Got {v}")
        return v
