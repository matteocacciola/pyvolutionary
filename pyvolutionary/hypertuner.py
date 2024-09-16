import operator
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial, reduce
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import concurrent.futures as parallel

from .abstract import OptimizationAbstract
from .enums import ModeSolver, TaskType, ExportType
from .models import Task, Agent, OptimizationResult


class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid: dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    See Also
    --------
    HyperTuner: Uses :class:`ParameterGrid` to perform a full parallelized parameter search.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(
                f"Parameter grid should be a dict or a list, got: {param_grid!r} of"
                f" type {type(param_grid).__name__}"
            )

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(
                        f"Parameter array for {key!r} should be one-dimensional, got:"
                        f" {value!r} with shape {value.shape}"
                    )
                if isinstance(value, str) or not isinstance(
                    value, (np.ndarray, Sequence)
                ):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" numpy array, but got {value!r} (of type "
                        f"{type(value).__name__}) instead. Single values "
                        "need to be wrapped in a list with one element."
                    )
                if len(value) == 0:
                    raise ValueError(
                        f"Parameter grid for parameter {key!r} need "
                        f"to be a non-empty sequence, got: {value!r}"
                    )

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.prod can't).
        prd = partial(reduce, operator.mul)
        return sum(
            prd(len(v) for v in p.values()) if p else 1 for p in self.param_grid
        )

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                ind -= 1
                continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.prod(sizes)

            if ind < total:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out
            # Try the next grid
            ind -= total

        raise IndexError("ParameterGrid index out of range")


class HyperTuner:
    """
    HyperTuner utility.

    This is a feature that enables the tuning of hyperparameters for an algorithm.
    It also supports exporting results in various formats, such as Pandas DataFrame, JSON, and CSV.
    This feature provides a better option compared to using GridSearchCV or ParameterGrid from the scikit-learn library
    to tune hyperparameters

    The important functions to note are "execute()" and "resolve()".

    Args:
        algorithm (OptimizationAbstract): the algorithm/optimizer to tune
        param_grid (dict, list): dict or list of dictionaries
        n_trials (int): number of repetitions
        mode (str): set the mode to run (sequential, thread, process), default="sequential"
        n_workers (int): effected only when mode is "thread" or "process".
    """

    def __init__(self, algorithm: OptimizationAbstract, param_grid: dict | list = None, **kwargs: object) -> None:
        self.__set_keyword_arguments__(kwargs)
        self._algorithm = algorithm
        self._param_grid = param_grid

        self._problem: Task | None = None
        self._results = None
        self._debug: bool | None = None

        self._best_row: pd.DataFrame | None = None
        self._best_params, self._best_score = None, None
        self._df_fit: pd.DataFrame | None = None
        self._df_loss: pd.DataFrame | None = None

    def __set_keyword_arguments__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def best_parameters(self) -> dict:
        return self._best_params

    @best_parameters.setter
    def best_parameters(self, x):
        self._best_params = x

    @property
    def best_row(self) -> dict:
        return self._best_row.to_dict()

    @property
    def best_score(self) -> float:
        return self._best_score

    def export_results(self, save_as: str, save_path: str | None = None):
        """
        Export results to various file type
        :param save_as: the file name (with file type, e.g. dataframe, json, csv) that hold results
        :param save_path: the path to the folder, default: "best_fit/{algorithm_name}"
        :raises: ValueError: raises ValueError if export type is not supported
        """
        if save_as not in ExportType:
            raise ValueError(f"Export type {save_as} is not supported")

        # check parent directories
        save_path = save_path if save_path is not None else f"best_fit/{self._algorithm.name}"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        filename = f"tuning_best_fit_{self._algorithm.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if save_as == str(ExportType.JSON):
            self._df_fit.to_json(f"{save_path}/{filename}.json")
            return
        if save_as == str(ExportType.DATAFRAME):
            self._df_fit.to_pickle(f"{save_path}/{filename}.pkl")
            return
        self._df_fit.to_csv(f"{save_path}/{filename}.csv", header=True, index=False)

    def __run__(self, id_trial: int, mode: ModeSolver, n_workers: int | None = None) -> tuple[int, Agent, list]:
        result = self._algorithm.optimize(self._problem, mode=str(mode), workers=n_workers)
        return id_trial, result.best_solution, result.rates

    @staticmethod
    def __generate_dict_result__(params, trial, loss_list):
        result_dict = dict(params)
        result_dict["trial"] = trial
        keys = np.arange(1, len(loss_list) + 1)
        result_dict = {**result_dict, **dict(zip(keys, loss_list))}
        return result_dict

    def execute(
        self,
        task: Task,
        n_trials: int | None = 2,
        n_jobs: int | None = 2,
        mode: str | None = "serial",
        n_workers: int | None = 2,
        debug: bool | None = False,
    ) -> None:
        """
        Execute Tuner utility
        :param task: the task to solve
        :param n_trials: number of trials on the Task, default=2
        :param n_jobs: number of jobs to run to speed this task up (run multiple trials at the same time), default=2,
            max = os.cpu_count() - 1
        :param mode: apply on current Task ("serial", "thread", "process"), default="serial"
        :param n_workers: apply on current Task, number of processes if mode is "thread" or "process", default=2
        :param debug: switch for verbose logging, default=False
        :raises: ValueError: raises ValueError if mode is not supported
        """
        self._problem = task
        self._debug = debug
        try:
            mode = ModeSolver(mode)
        except ValueError:
            raise ValueError("Invalid mode. Possible values are \"serial\", \"thread\" and \"process\"")

        n_cpus = np.clip(n_jobs, 2, os.cpu_count() - 1, dtype=int)

        list_params_grid = list(ParameterGrid(self._param_grid))
        trial_columns = [f"trial_{id_trial}" for id_trial in range(1, n_trials + 1)]
        ascending = True if self._problem.minmax == TaskType.MIN else False

        best_fit_results = []
        loss_results = []
        for id_params, params in enumerate(list_params_grid):
            self._algorithm.set_config_parameters(params)
            best_fit_results.append({"params": params})
            with parallel.ProcessPoolExecutor(n_cpus) as executor:
                list_results = executor.map(
                    partial(self.__run__, n_workers=n_workers, mode=mode), list(range(0, n_trials))
                )
                for (idx, g_best, loss_epoch) in list_results:
                    best_fit_results[-1][trial_columns[idx]] = g_best.cost
                    loss_results.append(self.__generate_dict_result__(params, idx, loss_epoch))

                    self.__debug_results__(params, idx, g_best)

        self._df_fit = pd.DataFrame(best_fit_results)
        self._df_fit["trial_mean"] = self._df_fit[trial_columns].mean(axis=1)
        self._df_fit["trial_std"] = self._df_fit[trial_columns].std(axis=1)
        self._df_fit["rank_mean"] = self._df_fit["trial_mean"].rank(ascending=ascending)
        self._df_fit["rank_std"] = self._df_fit["trial_std"].rank(ascending=ascending)
        self._df_fit["rank_mean_std"] = self._df_fit[["rank_mean", "rank_std"]].apply(tuple, axis=1).rank(
            method="dense", ascending=ascending
        )
        self._best_row = self._df_fit[self._df_fit["rank_mean_std"] == self._df_fit["rank_mean_std"].min()]
        self._best_params = self._best_row["params"].values[0]
        self._best_score = self._best_row["trial_mean"].values[0]
        self._df_loss = pd.DataFrame(loss_results)
    
    def __debug_results__(self, params, idx: int, g_best: Agent):
        if self._debug:
            print(
                f"Algorithm: {self._algorithm.name}, params: {params}, trial: {idx + 1}, best cost: {g_best.cost}"
            )

    def resolve(self, mode: str = "serial", n_workers: int = None) -> OptimizationResult:
        """
        Resolving the problem with the best parameters
        :param mode: it can be "process" or "thread" for parallel execution, "sequential" otherwise. Specifically:
            "process", the parallel mode with multiple cores run the tasks; "thread", the parallel mode with multiple
            threads run the tasks; "sequential", the sequential mode that effect on updating phase of other agents
            (default: "sequential")
        :param n_workers: The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
        :return: the result of the optimization
        :rtype: OptimizationResult
        """
        self._algorithm.set_config_parameters(self.best_parameters)
        return self._algorithm.optimize(task=self._problem, mode=mode, workers=n_workers)
