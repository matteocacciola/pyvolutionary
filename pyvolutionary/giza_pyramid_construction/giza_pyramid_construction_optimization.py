import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Worker, GizaPyramidConstructionOptimizationConfig


class GizaPyramidConstructionOptimization(OptimizationAbstract):
    """
    Implementation of the Fire Hawk Optimization algorithm.

    Args:
        config (GizaPyramidConstructionOptimizationConfig): an instance of GizaPyramidConstructionOptimizationConfig class.
            {parse_obj_doc(FireHawkOptimizationConfig)}

    Bibliography
    ----------
    [1] Harifi, S., Mohammadzadeh, J., Khalilian, M. et al. Giza Pyramids Construction: an ancient-inspired
        metaheuristic algorithm for optimization. Evol. Intel. 14, 1743–1761 (2021).
        https://doi.org/10.1007/s12065-020-00451-3
    """
    def __init__(self, config: GizaPyramidConstructionOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        mu1, mu2 = self._config.friction
        g = 2 * self._config.gravity
        theta = np.deg2rad(self._config.theta)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        pss = self._config.prob_substitution

        dim = self._task.space_dimension
        dims = list(range(0, dim))

        candidates = []
        for worker in self._population:
            pos = np.array(worker.position)

            v0 = np.random.random() ** 2
            mu = np.random.uniform(mu1, mu2)
            d = v0 / (g * (sin_theta + (mu * cos_theta)))  # stone destination
            x = v0 / (g * sin_theta)  # worker movement

            new_pos = self._uniform_position() * x * (pos + d)

            # substitution
            first_condition = dims == np.repeat(np.random.randint(0, dim), dim)
            second_condition = np.random.random(dim) <= pss
            new_pos = self._correct_position(
                np.where(np.logical_or(first_condition, second_condition), new_pos, pos)
            )

            new_agent = Worker(**self._init_agent(new_pos).model_dump())

            if new_agent.cost < worker.cost:
                candidates.append(new_agent)

        self._extend_and_trim_population(candidates)
