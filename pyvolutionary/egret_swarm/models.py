import numpy as np

from ..models import Agent, BaseOptimizationConfig


class Egret(Agent):
    m: list[float]
    v: list[float]
    weights: list[float]
    g: list[float] | None = None

    def __init__(self, **kwargs):
        # kwargs["local_position"] = kwargs.get("local_position", kwargs["position"])
        # kwargs["local_cost"] = kwargs.get("local_cost", kwargs["cost"])
        weights = np.array(kwargs["weights"])
        position = np.array(kwargs["position"])
        cost = kwargs["cost"]
        kwargs["g"] = ((np.sum(weights * position) - cost) * position).tolist()
        super().__init__(**kwargs)


class EgretSwarmOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Egret Swarm Optimization algorithm.
    """
    pass
