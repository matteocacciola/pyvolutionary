from datetime import datetime
from itertools import chain
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import concurrent.futures as parallel
from copy import deepcopy
import os

from .models import Task
from .abstract import OptimizationAbstract
from .enums import ModeSolver, ExportType


class Multitask:
    """
    Multitask utility class.

    This feature enables the execution of multiple algorithms across multiple problems and trials.
    Additionally, it allows for exporting results in various formats such as Pandas DataFrame, JSON, and CSV.

    Args:
        algorithms (list, tuple): List of algorithms to run
        tasks (list, tuple): List of problems to run
        modes (list, tuple): List of modes to apply on algorithm/problem
        n_workers (int): Number of workers (threads or processes) to apply on algorithm/problem. Only effect when `mode`
            is `thread` or `process`
    """
    def __init__(
        self,
        algorithms: tuple,
        tasks: tuple,
        modes: tuple[str] | None = None,
        n_workers: int | None = None,
        **kwargs: object
    ) -> None:
        self.__set_keyword_arguments__(kwargs)
        self._algorithms = algorithms
        self._tasks = tasks
        self._n_algorithms = len(self._algorithms)
        self._m_tasks = len(self._tasks)
        self._modes = self.__check_input__("modes", "str (thread, process, serial)", modes)
        self._n_workers = n_workers
        self._debug: bool | None = None
        self._df2: list[pd.DataFrame] = []

        self.__check_modes__()

    def __check_input__(self, name: str, kind: str, values: tuple | None = None):
        if values is None:
            return None
        if not isinstance(values, tuple):
            raise ValueError(f"{name} should be a tuple of {kind} instances.")

        if len(values) == 1:
            return [[deepcopy(values[0]) for _ in range(0, self._m_tasks)] for _ in range(0, self._n_algorithms)]
        if len(values) == self._n_algorithms:
            return [deepcopy(values[idx] for _ in range(0, self._m_tasks)) for idx in range(0, self._n_algorithms)]
        if len(values) == self._m_tasks:
            return [deepcopy(values) for _ in range(0, self._n_algorithms)]
        if len(values) == (self._n_algorithms * self._m_tasks):
            return values

        raise ValueError(f"{name} should be list of {kind} instances with size (1) or (n) or (m) or (n*m), "
                         f"where n is #algorithms, m is #problems.")

    def __check_modes__(self):
        if self._modes is None:
            return

        are_in_mode_solver = all([mode in ModeSolver for mode in list(chain.from_iterable(self._modes))])
        if not are_in_mode_solver:
            raise ValueError(f"Invalid mode. Possible values are \"serial\", \"thread\" and \"process\"")

    def __set_keyword_arguments__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def export_to_dataframe(result: pd.DataFrame, save_path: str):
        result.to_pickle(f"{save_path}.pkl")

    @staticmethod
    def export_to_json(result: pd.DataFrame, save_path: str):
        result.to_json(f"{save_path}.json")

    @staticmethod
    def export_to_csv(result: pd.DataFrame, save_path: str):
        result.to_csv(f"{save_path}.csv", header=True, index=False)

    def export_results(self, save_as: str, save_path: str | None = None):
        """
        Export results to various file type
        :param save_as: the file name (with file type, e.g. dataframe, json, csv) that hold results
        :param save_path: the path to the folder, default: "multitask", with subfolders for each algorithm
        :raises: ValueError: raises ValueError if export type is not supported
        """
        if save_as not in ExportType:
            raise ValueError(f"Export type {save_as} is not supported")

        export_function = getattr(self, f"export_to_{save_as}")

        # check parent directories
        for id_optimizer, optimizer in enumerate(self._algorithms):
            save_path = save_path if save_path is not None else "multitask"
            save_path = f"{save_path}/{optimizer.name}"
            Path(save_path).mkdir(parents=True, exist_ok=True)

            filename = f"tuning_best_fit_{optimizer.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            export_function(self._df2[id_optimizer], f"{save_path}/{filename}")

    def __run__(
        self,
        id_trial: int,
        optimizer: OptimizationAbstract,
        task: Task,
        mode: ModeSolver
    ) -> dict:
        result = optimizer.optimize(task, mode=str(mode), workers=self._n_workers)
        return {
            "id_trial": id_trial,
            "solution": result,
            "problem_name": task.name,
        }

    def execute(
        self,
        n_trials: int = 2,
        n_jobs: int = 2,
        debug: bool = False
    ) -> None:
        """
        Execute multitask utility.
        :param n_trials: number of trials on the Task, default=2
        :param n_jobs: number of jobs to run to speed this task up (run multiple trials at the same time), default=2,
            max = os.cpu_count() - 1
        :param debug: switch for verbose logging, default=False
        :raises: ValueError: raises ValueError if any of the modes is not supported
        """
        self._debug = debug
        n_cpus = np.clip(n_jobs, 2, os.cpu_count() - 1, dtype=int)
        trial_list = list(range(1, n_trials + 1))

        for id_optimizer, optimizer in enumerate(self._algorithms):
            best_fit_optimizer_results = {}
            for id_task, task in enumerate(self._tasks):
                mode = self.__get_mode__(id_optimizer, id_task)

                best_fit_trials = self.__parallelize__(optimizer, task, mode, n_cpus, trial_list)

                best_fit_optimizer_results[f"{optimizer.name}_{task.name}"] = best_fit_trials

            self._df2.append(pd.DataFrame(best_fit_optimizer_results))

    def __parallelize__(
        self, optimizer: OptimizationAbstract, task: Task, mode: ModeSolver, n_cpus: int, trial_list: list
    ) -> list:
        best_fit_trials = []
        with parallel.ProcessPoolExecutor(n_cpus) as executor:
            list_results = executor.map(partial(self.__run__, optimizer=optimizer, task=task, mode=mode), trial_list)
            for result in list_results:
                best_fit_trials.append(result)
                self.__debug_results__(result, optimizer.name)
        return best_fit_trials

    def __get_mode__(self, id_optimizer: int, id_prob: int) -> ModeSolver:
        mode = "serial"
        if self._modes is not None:
            mode = self._modes[id_optimizer][id_prob]
        try:
            mode = ModeSolver(mode)
        except ValueError:
            raise ValueError("Invalid mode. Possible values are \"serial\", \"thread\" and \"process\"")
        return mode

    def __debug_results__(self, result: dict, optimizer_name: str):
        if self._debug:
            print(f"Solving task {result['problem_name']} using algorithm {optimizer_name} on the "
                  f"{result['id_trial']} trial")
