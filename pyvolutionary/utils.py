from .helpers import sort_by_cost
from .models import OptimizationResult


def agent_trend(result: OptimizationResult, idx: int, iters: list[int] | None = None) -> list[float]:
    """
    This method returns the trend of the given agent's cost across the specified iterations. If iters is None, then the
    trend of agent's cost is considered for the entire evolution of the agent itself during the optimization process.
    :param result: the result of the optimization
    :param idx: the index of the agent
    :param iters: the iterations
    :return: the trend of the agent's cost
    :rtype: list[float]
    """
    if iters is None:
        iters = range(0, len(result.evolution))

    return [sort_by_cost(result.evolution[i].agents)[idx].cost for i in iters]


def best_agent_trend(result: OptimizationResult, iters: list[int] | None = None) -> list[float]:
    """
    This method returns the trend of the best agent's cost across the specified iterations. If iters is None, then the
    trend of best agent's cost is considered for the entire evolution of the agent itself during the optimization
    process.
    :param result: the result of the optimization
    :param iters: the iterations
    :return: the trend of the best agent's cost
    :rtype: list[float]
    """
    return agent_trend(result, 0, iters)


def agent_position(result: OptimizationResult, idx: int, iters: list[int] | None = None) -> list[list[float | int]]:
    """
    This method returns the location of the considered agent in the search domain across the specified iterations. If
    iters is None, then the trend of agent's position is considered for the entire evolution of the agent itself during
    the optimization process.
    :param result: the result of the optimization
    :param idx: the index of the agent
    :param iters: the iterations
    :return: the evolution of the agent's position in the search space
    :rtype: list[list[float | int]]
    """
    if iters is None:
        iters = range(0, len(result.evolution))

    return [sort_by_cost(result.evolution[i].agents)[idx].position for i in iters]


def best_agent_position(result: OptimizationResult, iters: list[int] | None = None) -> list[list[float | int]]:
    """
    This method returns the location of the best agent in the search domain across the specified iterations. If iters is
    None, then the trend of best agent's position is considered for the entire evolution of the best agent itself during
    the optimization process.
    :param result: the result of the optimization
    :param iters: the iterations
    :return: the evolution of the agent's position in the search space
    :rtype: list[list[float | int]]
    """
    return agent_position(result, 0, iters)
