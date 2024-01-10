from typing import Final
import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Firework, FireworksOptimizationConfig


class FireworksOptimization(OptimizationAbstract):
    """
    Implementation of the Fireworks Algorithm Optimization algorithm.

    Args:
        config (FireworksOptimizationConfig): an instance of FireworksAlgorithmOptimizationConfig class.
            {parse_obj_doc(FireworksOptimizationConfig)}

    Bibliography
    ----------
    [1] Yang, X. S. (2010). Fireworks algorithm for optimization. International Journal of Bio-Inspired Computation,
        2(4), 78–84. https://doi.org/10.1504/IJBIC.2010.037285
    [2] Yang, X. S. (2010). Fireworks Algorithm: A Novel Swarm Intelligence Optimization Method. In Nature Inspired
        Cooperative Strategies for Optimization (pp. 355–364). Springer Berlin Heidelberg.
        https://doi.org/10.1007/978-3-642-12538-6_35
    [3] Yang, X. S. (2012). Fireworks Algorithm for Optimization. In Nature-Inspired Optimization Algorithms
        (pp. 355–364). Elsevier. https://doi.org/10.1016/B978-0-12-416743-8.00016-2
    [4] Yang, X. S. (2013). Fireworks Algorithm: Recent Advances and Applications. In Studies in Computational
        Intelligence (Vol. 476, pp. 355–364). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-31688-4_17
    [5] Yang, X. S. (2014). Fireworks Algorithm: A Novel Swarm Intelligence Optimization Method. In Nature-Inspired
        Computation in Engineering (pp. 355–364). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-39829-2_35
    [6] Yang, X. S. (2015). Fireworks Algorithm: A Survey. International Journal of Swarm Intelligence Research,
        5(1), 1–18. https://doi.org/10.4018/IJSIR.2015010101
    """
    EPS: Final[float] = np.finfo(float).eps

    def __init__(self, config: FireworksOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)

    def optimization_step(self):
        def get_num_sparks(firework: Firework) -> int:
            si = sparks_num * (costs[-1] - firework.cost + self.EPS) / (pop_size * costs[-1] - np.sum(costs) + self.EPS)
            if si < a * sparks_num:
                return int(round(a * sparks_num) + 1)
            if si > b * sparks_num:
                return int(round(b * sparks_num) + 1)
            return int(round(si) + 1)

        def explode(firework: Firework) -> Firework:
            Ai = explosion_amplitude * (firework.cost - costs[0] + self.EPS) / (np.sum(costs) - costs[0] + self.EPS)
            pos_new = firework.position
            list_idx = np.random.choice(range(0, n_dims), round(np.random.uniform() * n_dims), replace=False)
            pos_new = [pos_new[i] + int(i in list_idx) * Ai * np.random.uniform(-1, 1) for i in range(0, n_dims)]
            return Firework(**self._init_agent(pos_new).model_dump())

        def get_subsparks() -> Firework:
            idx = np.random.randint(0, pop_size)
            pos_new = self._population[idx].position
            list_idx = np.random.choice(range(0, n_dims), round(np.random.uniform() * n_dims), replace=False)
            # Gaussian explosion
            pos_new = [pos_new[i] + int(i in list_idx) * np.random.normal(0, 1) for i in range(0, n_dims)]
            return Firework(**self._init_agent(pos_new).model_dump())

        sparks_num = self._config.sparks_num
        pop_size = self._config.population_size
        explosion_amplitude = self._config.explosion_amplitude
        a = self._config.a
        b = self._config.b

        n_dims = self._task.space_dimension

        costs = sorted([firework.cost for firework in self._population])

        # generate Gaussian sparks
        sparks = [explode(firework) for firework in self._population for _ in range(0, get_num_sparks(firework))] + [
            get_subsparks() for _ in range(0, sparks_num)
        ]

        self._extend_and_trim_population(sparks)

