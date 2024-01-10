import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Cell, BacterialForagingOptimizationConfig


class BacterialForagingOptimization(OptimizationAbstract):
    """
    Implementation of the Bacterial Foraging Optimization algorithm, in its Adaptive version.

    Args:
        config (BacterialForagingOptimizationConfig): an instance of BacterialForagingOptimization class.
            {parse_obj_doc(AdaptiveBacterialForagingOptimizationConfig)}

    Bibliography
    ----------
    [1] Passino, K.M., 2002. Biomimicry of bacterial foraging for distributed optimization and control.
        IEEE control systems magazine, 22(3), pp.52-67.
    [2] Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019, April. Building resource auto-scaler with functional-link
        neural network and adaptive bacterial foraging optimization. In International Conference on
        Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
    """
    def __init__(self, config: BacterialForagingOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__C_s: np.ndarray | None = None
        self.__C_e: np.ndarray | None = None

    def before_initialization(self):
        self.__C_s = self._config.C_s * self._bandwidth()
        self.__C_e = self._config.C_e * self._bandwidth()

    def _init_agent(self, position: list[float] | np.ndarray | None = None) -> Cell:
        agent = super()._init_agent(position)
        return Cell(**agent.model_dump(), local_best=agent.position.copy(), local_cost=agent.cost)

    def __clean_population__(self):
        """
        Remove duplicates from the population. This method is called after each optimization step.
        """
        new_set = set()
        pop = self._population.copy()
        for idx, obj in enumerate(pop):
            if pos := tuple(obj.position) in new_set:
                self._population.pop(idx)
            else:
                new_set.add(pos)

    def __balance_population__(self):
        """
        Balance the population by adding more agents or remove some agents. This method is called after each
        optimization step. The population size is fixed to the value of the configuration. If the population size
        is greater than the value of the configuration, then some agents are removed. Otherwise, new agents are added.
        """
        n_agents = len(self._population) - self._config.population_size
        if n_agents == 0:
            return

        if n_agents < 0:
            self._population += [self._init_agent() for _ in range(0, -n_agents)]
        else:
            list_idx_removed = np.random.choice(range(0, len(self._population)), n_agents, replace=False)
            for idx in sorted(list_idx_removed, reverse=True):
                self._population.pop(idx)

    def optimization_step(self):
        def update_step_size(c: Cell) -> float:
            step_sz = C_s - (C_s - C_e) * c.cost / total_costs
            return step_sz / c.nutrients if c.nutrients > 0 else step_sz

        def tumble_cell(c: Cell, step_sz: float) -> Cell:
            position = np.array(c.position)
            delta_i = (best_position - position) + (np.array(c.local_best) - position)
            delta = np.sqrt(np.dot(delta_i, delta_i.T))
            if delta == 0:
                delta_i = np.random.uniform(-1.0, 1.0, self._task.space_dimension)
                delta = np.sqrt(np.dot(delta_i, delta_i.T))
            return self._init_agent(c.position + step_sz * delta_i / delta)

        def swim(c: Cell, step_sz: float) -> Cell:
            for _ in range(Ns):
                # move the bacterium to the location of the new cell and evaluate the moved position
                new_cell = tumble_cell(c, step_sz)
                if new_cell.cost >= c.cost:
                    c.nutrients -= 1
                    continue
                new_cell.nutrients += 1
                c = new_cell
                # update personal best
                if new_cell.cost < c.local_cost:
                    c.local_best = new_cell.position.copy()
                    c.local_cost = new_cell.cost
            return c

        C_s, C_e = self.__C_s, self.__C_e
        Ns = self._config.Ns
        best_position = np.array(self._best_agent.position)

        # evolve the population
        for idx in range(0, self._config.population_size):
            cell = self._population[idx]
            total_costs = np.sum([a.cost for a in self._population])

            step_size = update_step_size(cell)
            cell = swim(cell, step_size)

            m = max(self._config.N_split, self._config.N_split + (
                len(self._population) - self._config.population_size
            ) / self._config.N_adapt)

            pos = np.array(cell.position)

            if cell.nutrients > m:
                tt = np.random.normal(0, 1, self._task.space_dimension)
                agent = self._init_agent(
                    self._correct_position(tt * pos + (1 - tt) * (np.array(self._best_agent.position) - pos))
                )
                self._population.append(agent)
            nut_min = min(self._config.N_adapt, self._config.N_adapt + (
                    len(self._population) - self._config.population_size
            ) / self._config.N_adapt)

            if cell.nutrients < nut_min and np.random.random() < self._config.Ped:
                self._population[idx] = self._init_agent()

        # make sure the population does not have duplicates
        self.__clean_population__()

        # balance the population by adding more agents or remove some agents
        self.__balance_population__()