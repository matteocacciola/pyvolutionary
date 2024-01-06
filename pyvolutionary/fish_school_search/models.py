import numpy as np

from ..models import Agent, BaseOptimizationConfig


class Fish(Agent):
    weight: float
    delta_pos: list[float] | None = None
    delta_cost: float | None = None


class FishSchoolSearchOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Fish School Search Optimization.\n
        step_individual_init (float): length of initial individual step.\n
        step_individual_final (float): length of final individual step.\n
        step_volitive_init (float): length of initial volatile step.\n
        step_volitive_final (float): length of final volatile step.\n
        min_w (float): minimum weight of a fish.\n
        w_scale (float): scale weight of a fish.
    """
    step_individual_init: float
    step_individual_final: float
    step_volitive_init: float
    step_volitive_final: float
    min_w: float
    w_scale: float
