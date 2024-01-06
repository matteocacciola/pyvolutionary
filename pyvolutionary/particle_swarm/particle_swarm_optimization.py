import numpy as np

from ..models import ContinuousVariable
from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Particle, ParticleSwarmOptimizationConfig


class ParticleSwarmOptimization(OptimizationAbstract):
    """
    Implementation of the Particle Swarm Optimization algorithm.

    Args:
        config (ParticleSwarmOptimizationConfig): an instance of ParticleSwarmOptimizationConfig class.
            {parse_obj_doc(ParticleSwarmOptimizationConfig)}

    Bibliography
    ----------
    [1] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN’95 - International
        Conference on Neural Networks, 4, 1942–1948 vol.4. https://doi.org/10.1109/ICNN.1995.488968
    [2] Kennedy, J., & Eberhart, R. C. (1995). A new optimizer using particle swarm theory. Proceedings of the Sixth
        International Symposium on Micro Machine and Human Science, 39–43. https://doi.org/10.1109/MHS.1995.494215
    3[] Eberhart, R. C., & Shi, Y. (2001). Particle swarm optimization: Developments, applications and resources.
        Proceedings of the 2001 Congress on Evolutionary Computation (IEEE Cat. No.01TH8546), 1, 81–86.
        https://doi.org/10.1109/CEC.2001.934374
    """
    def __init__(self, config: ParticleSwarmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__v: np.ndarray | None = None
        self.__pbest: list[Particle] = []

    def before_initialization(self):
        self.__v = 0.2 * self._bandwidth()

    def _init_agent(
        self,
        position: list[float] | np.ndarray | None = None,
        velocity: list[float] | np.ndarray | None = None
    ) -> Particle:
        agent = super()._init_agent(position=position)

        if velocity is None:
            return Particle(**agent.model_dump(), velocity=np.zeros(self._task.space_dimension).tolist())

        velocity = np.where(
            isinstance(self._task.variables, ContinuousVariable), np.clip(velocity, -self.__v, self.__v), velocity
        ).tolist()
        return Particle(**agent.model_dump(), velocity=velocity)

    def _init_population(self):
        super()._init_population()
        self.__pbest = self._population.copy()

    def optimization_step(self):
        """
        Merge the population and the archive, sort them by cost and update the population with the best n_ants
        """
        # Update the particles
        w_min, w_max = self._config.w

        for idx in range(0, self._config.population_size):
            particle = self._population[idx].model_copy()
            pbest = self.__pbest[idx].model_copy()

            # Randomly generate r1, r2 and inertia weight from normal distribution, and then calculate the new velocity
            w = np.random.uniform(w_min, w_max)  # Inertia weight * (cycles + 1) / max_cycles

            velocity = w * np.array(particle.velocity) + self._config.c1 * np.random.rand() * (
                np.array(pbest.position) - np.array(particle.position)
            ) + self._config.c2 * np.random.rand() * (
                    np.array(self._best_agent.position) - np.array(particle.position)
            )

            # Move particles by adding velocity
            self._population[idx] = self._init_agent(position=particle.position + velocity, velocity=velocity)

            # Update personal best
            self.__pbest[idx] = self._greedy_select_agent(self._population[idx], pbest)
