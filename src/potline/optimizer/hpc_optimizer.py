"""
Custom optimizer class for XPOT, based on the skopt.Optimizer class.
"""

from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path
from typing import Generic, TypeVar, Union, Callable

import skopt # type: ignore
from skopt import plots # type: ignore
from matplotlib import pyplot as plt
from tabulate import tabulate

DictValue = str | list[str | int] | int | float
NestedDict = dict[str, Union["NestedDict", DictValue]]

Key = TypeVar("Key")

_exec_path = Path(os.getcwd()).resolve()

class HPCOptimizer(Generic[Key]):
    """
    The HPCOptimizer class is a wrapper around the skopt.Optimizer class
    that allows for the use of named parameters, and implements the
    ask-tell interface, dump and load methods, and result recording.

    This class can be used to initialise any optimizer to be used by XPOT
    for optimizing hyperparameters for fitting ML potentials. This class is
    used for all classes

    Parameters
    ----------
    optimisable_params : dict
        Dictionary of parameter names and skopt.space.Dimension objects.
    sweep_path : str
        Path of directory to save files to.
    n_points : int, optional
        Number of points to ask for, by default 1
    strategy : str, optional
        Strategy to use for the optimisation, by default "cl_min"
    skopt_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the skopt.Optimiser
        class, by default None. You should define any non-default parameters
    """
    def __init__(
        self,
        optimisable_params: dict[Key, skopt.space.Dimension],
        sweep_path: str,
        n_points: int = 1,
        strategy: str = "cl_min",
        skopt_kwargs: dict[str, str | int | float] | None = None,
    ):
        self._optimiser = skopt.Optimizer(
            dimensions=list(optimisable_params.values()),
            random_state=42,
            **skopt_kwargs,
        )
        self.sweep_path = sweep_path
        self.n_points = n_points
        self.strategy = strategy
        self._optimisable_params = optimisable_params
        self.initialise_csvs(sweep_path)
        self.iter = 1
        self.subiter = 1

    def ask(self) -> list[dict[Key, DictValue]]:
        """
        Ask the optimizer for a new set of parameters based on the current
        results of the optimisation.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        param_values_list: list[list[float | int | str]] = \
            self._optimiser.ask(self.n_points, self.strategy)
        return [dict(zip(self._optimisable_params.keys(), param_values))
                for param_values in param_values_list]

    def tell(self, params_list: list[dict[Key, DictValue]], results_list: list[float]) -> None:
        """
        Tell the optimiser the result of the last iteration, as well as the
        parameter values used to achieve it.

        Parameters
        ----------
        params_list : list
            List of dictionaries of parameter names and values.
        results_list : list
            List of results from the last iteration.
        """
        # 1. make sure that we get the order correct
        locations_list = [[params[name] for name in self._optimisable_params] for params in params_list]
        # 2. tell the optimiser
        self._optimiser.tell(locations_list, results_list)

    def dump_optimiser(self, path: str) -> None:
        """
        Dump the optimiser to a file.

        Parameters
        ----------
        path : str
            Path of directory to write file to.
        """
        with open(f"{path}/xpot-optimiser.pkl", "wb") as f:
            pickle.dump(self._optimiser, f)

    def load_optimiser(self, path: str) -> None:
        """
        Load the optimiser from a file.

        Parameters
        ----------
        path : str
            File path.
        """
        with open(path, "rb") as f:
            self._optimiser = pickle.load(f)

        if len(self._optimiser.space.dimensions) != len(
            self._optimisable_params
        ):
            raise ValueError(
                "The optimiser and the optimisable parameters "
                "have different lengths. The optimiser cannot be "
                "loaded."
            )

    def initialise_csvs(self, path: str) -> None:
        """
        Initialise the CSV files for the optimisation.

        Parameters
        ----------
        path : str
            Path of directory to save files to.
        """
        keys = [" ".join(i[0]) for i in self._optimisable_params] # type: ignore
        with open(f"{path}/parameters.csv", "w+", encoding='utf-8') as f:
            f.write("iteration,subiteration,loss," + ",".join(map(str, keys)) + "\n")
        with open(f"{path}/atomistic_errors.csv", "w+", encoding='utf-8') as f:
            f.write(
                "Iteration,"
                + "Subiteration,"
                + "Train Δ Energy,"
                + "Test Δ Energy,"
                + "Train Δ Force,"
                + "Test Δ Force"
                + "\n"
            )
        with open(f"{path}/loss_function_errors.csv", "w+", encoding='utf-8') as f:
            f.write(
                "Iteration,"
                + "Subiteration,"
                + "Train Δ Energy,"
                + "Test Δ Energy,"
                + "Train Δ Force,"
                + "Test Δ Force"
                + "\n"
            )
        print("Initialised CSV Files")

    def write_param_result(
        self,
        path: str,
        iteration: int,
        subiteration: int,
    ) -> None:
        """
        Write the current iteration and loss to the parameters.csv file.

        Parameters
        ----------

        path : str
            Path of directory with parameters.csv file.
        iteration : int
            Current iteration.
        loss : float
            Loss of current iteration.
        params : dict
            List of parameter values for the current iteration.
        """
        # Raise error if there is no parameters.csv file
        if not os.path.isfile(f"{path}/parameters.csv"):
            raise FileNotFoundError(
                f"parameters.csv file does not exist at {path}"
            )
        i: int = self.n_points - subiteration + 1
        with open(f"{path}/parameters.csv", "a", encoding='utf-8') as f:
            f.write(
                f"{iteration},"
                + f"{subiteration},"
                + f"{self._optimiser.yi[-i]},"
                + ",".join([str(i) for i in self._optimiser.Xi[-i]])
                + "\n"
            )
        print(f"Iteration {iteration}.{subiteration} written to parameters.csv")

    def tabulate_final_results(
        self,
        path: str,
    ) -> None:
        """
        Tabulate the final results of the optimisation into pretty tables, with
        filenames

        Parameters
        ----------
        path : str
            Path of directory for all error files.
        """

        def tabulate_csv(file):
            with open(f"{file}.csv", encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)
                table = tabulate(rows, headers="firstrow", tablefmt="github")
            with open(f"{file}_final", "a+", encoding='utf-8') as f:
                f.write(table)

        tabulate_csv(f"{path}/parameters")
        tabulate_csv(f"{path}/atomistic_errors")
        tabulate_csv(f"{path}/loss_function_errors")

    def run_hpc_optimization(
        self,
        dispatcher: Callable,
        collector: Callable,
        path=_exec_path,
        **kwargs,
    ) -> None:
        """
        Function for running optimisation sweep.

        Parameters
        ----------

        objective : callable
            Function to be optimised. Must return a float (loss value).
        path : str, optional
            Path of directory to get files from, by default "./".
        **kwargs
            Keyword arguments to pass to objective function.

        Returns
        -------
        loss
            Loss value of the current iteration.
        """
        next_params_list: list[dict[Key, DictValue]] = self.ask()
        fit_trackers: list[FitJobTracker] = []
        for next_params in next_params_list:
            fit_trackers.append(FitJobTracker(
                job_id=dispatcher(next_params, iteration=self.iter, subiter=self.subiter,**kwargs),
                iteration=self.iter, subiter=self.subiter, params=next_params, loss=0.0))
            self.subiter += 1
        for fit_tr in fit_trackers:
            fit_tr.loss = collector(fit_tr.job_id, fit_tr.iteration, fit_tr.subiter)

        self.tell([fit_tr.params for fit_tr in fit_trackers], [fit_tr.loss for fit_tr in fit_trackers])
        for fit_tr in fit_trackers:
            self.write_param_result(path, fit_tr.iteration, fit_tr.subiter)

        self.subiter = 1
        self.iter += 1

    def plot_results(self, path: str) -> None:
        """
        Function to create scikit-optimize results using inbuilt functions.

        Parameters
        ----------
        path : str
            Path of directory to save plots to.
        """
        data = self._optimiser.get_result()
        a = plots.plot_objective(data, levels=20, size=3)
        plt.tight_layout()
        a.figure.savefig(f"{path}/objective.pdf")

        b = plots.plot_evaluations(data)
        b.figure.savefig(f"{path}/evaluations.pdf")

class FitJobTracker(Generic[Key]):
    """
    Class to track the progress of a job in the optimisation sweep.

    Parameters
    ----------
    job_id : int
        Job ID of the current job.
    iteration : int
        Current iteration.
    subiter : int
        Current subiteration.
    params : dict
        Dictionary of parameter names and values.
    loss : float
        Loss value of the current iteration.
    """
    def __init__(self, job_id: int, iteration: int, subiter: int,
                 params: dict[Key, DictValue], loss: float) -> None:
        self.job_id = job_id
        self.iteration = iteration
        self.subiter = subiter
        self.params = params
        self.loss = loss
