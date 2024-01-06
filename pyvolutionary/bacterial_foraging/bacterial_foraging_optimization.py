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

    def __update_step_size__(self, cell: Cell) -> float:
        """
        Update the step size of the cell. The step size is computed as follows:\n
        s / cell.nutrients if cell.nutrients > 0 else s, where:\n
        s = C_s - (C_s - C_e) * cell.cost / total_costs\n
        :param cell: the cell to update the step size.
        :return: the step size.
        :rtype: float
        """
        total_costs = np.sum([agent.cost for agent in self._population])
        step_size = self.__C_s - (self.__C_s - self.__C_e) * cell.cost / total_costs
        return step_size / cell.nutrients if cell.nutrients > 0 else step_size

    def __tumble_cell__(self, cell: Cell, step_size: float) -> Cell:
        """
        Tumble the cell. The new position is computed as follows:\n
        new_position = cell.position + step_size * delta_i / delta, where:\n
        delta_i = (best_agent.position - cell.position) + (cell.local_best - cell.position)\n
        delta = sqrt(delta_i * delta_i.T)\n
        :param cell: the cell to tumble.
        :param step_size: the step size.
        :return: the new cell.
        :rtype: Cell
        """
        position = np.array(cell.position)
        delta_i = (np.array(self._best_agent.position) - position) + (np.array(cell.local_best) - position)
        delta = np.sqrt(np.dot(delta_i, delta_i.T))
        if delta == 0:
            delta_i = np.random.uniform(-1.0, 1.0, self._task.space_dimension)
            delta = np.sqrt(np.dot(delta_i, delta_i.T))

        return self._init_agent(self._correct_position(cell.position + step_size * delta_i / delta))

    def __swim__(self, cell: Cell, step_size: float) -> Cell:
        """
        Swim the cell. The cell will swim to a new cell or tumble some every time interval.
        :param cell: the cell to swim.
        :param step_size: the step size.
        :return: the cell after swimming.
        :rtype: Cell
        """
        for m in range(self._config.Ns):
            # move the bacterium to the location of the new cell and evaluate the moved position
            new_cell = self.__tumble_cell__(cell, step_size)
            if new_cell.cost >= cell.cost:
                cell.nutrients -= 1
                continue

            new_cell.nutrients += 1
            cell = new_cell
            # update personal best
            if new_cell.cost < cell.local_cost:
                cell.local_best = new_cell.position.copy()
                cell.local_cost = new_cell.cost

        return cell

    def __clean_population__(self):
        """
        Remove duplicates from the population. This method is called after each optimization step.
        """
        new_set = set()
        for idx, obj in enumerate(self._population):
            pos = tuple(obj.position)
            if pos in new_set:
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
        for idx, cell in enumerate(self._population):
            step_size = self.__update_step_size__(cell)
            cell = self.__swim__(cell, step_size)

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

            self._population[idx] = cell
            if cell.nutrients < nut_min and np.random.random() < self._config.Ped:
                self._population[idx] = self._init_agent()

        # make sure the population does not have duplicates
        self.__clean_population__()

        # balance the population by adding more agents or remove some agents
        self.__balance_population__()