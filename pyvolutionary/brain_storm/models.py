from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Person(Agent):
    pass


class ImprovedBrainStormOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Improved Brain Storm Optimization algorithm.
        m_clusters (int): [3, 10], number of clusters (m in the paper).\n
        p1 (float): (0, 1.0), 25% percent.\n
        p2 (float): (0, 1.0), 50% percent changed by its own (local search), 50% percent changed by outside (global search).\n
        p3 (float): (0, 1.0), 75% percent develop the old idea, 25% invented new idea based on levy-flight.\n
        p4 (float): (0, 1.0), need more weights on the centers instead of the random position.
    """
    m_clusters: int
    p1: float
    p2: float
    p3: float
    p4: float

    @field_validator("m_clusters")
    def correct_m_clusters(cls, v):
        if not 3 <= v <= 10:
            raise ValueError(f"\"m_clusters\" must be an integer in [3, 10]. Got {v}")
        return v

    @field_validator("p1", "p2", "p3", "p4")
    def correct_p(cls, v):
        if not 0 < v < 1.0:
            raise ValueError(f"\"p1\", \"p2\", \"p3\" and \"p4\" must be a float in (0, 1.0). Got {v}")
        return v


class BrainStormOptimizationConfig(ImprovedBrainStormOptimizationConfig):
    """
    Configuration class of the Brain Storm Optimization algorithm.
        m_clusters (int): [3, 10], number of clusters (m in the paper).\n
        p1 (float): (0, 1.0), 25% percent.\n
        p2 (float): (0, 1.0), 50% percent changed by its own (local search), 50% percent changed by outside (global search).\n
        p3 (float): (0, 1.0), 75% percent develop the old idea, 25% invented new idea based on levy-flight.\n
        p4 (float): (0, 1.0), need more weights on the centers instead of the random position.\n
        slope (int): [10, 50], changing logsig() function's slope (k: in the paper).
    """
    slope: int

    @field_validator("slope")
    def correct_slope(cls, v):
        if not 10 <= v <= 50:
            raise ValueError(f"\"slope\" must be an integer in [10, 50]. Got {v}")
        return v
