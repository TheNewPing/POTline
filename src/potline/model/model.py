"""
Base class for MLPs in XPOT using HPC.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

import xpot.loaders as load # type: ignore
from xpot import maths # type: ignore

from ..dispatcher import Dispatcher

class ColKW(Enum):
    """
    Column keywords.
    """
    ENERGY = "energies"
    FORCES = "forces"
    ATOMS = "atom_counts"

class PotModel(ABC):
    """
    Base class for MLIAP models.

    Args:
        - config_filepath: path to the configuration file.
        - default_filepath: path to the default configuration file.
        - out_path: path to the output directory.
        - energy_weight: weight of the energy loss.
        - hpc: flag to run the simulations on HPC.
    """
    def __init__(self, config_filepath: Path,
                 out_path: Path,
                 energy_weight: float,
                 hpc: bool,
                 default_filepath: Path):
        defaults = load.get_defaults(str(default_filepath))
        hypers = load.get_input_hypers(str(config_filepath))
        self.energy_weight: float = energy_weight
        self.hpc: bool = hpc
        self.out_path: Path = out_path
        self.out_path.mkdir(parents=True, exist_ok=False)

        self.mlp_total = load.merge_hypers(defaults, hypers)
        load.validate_subdict(self.mlp_total, hypers)
        self.optimisation_space = load.get_optimisable_params(self.mlp_total)
        self.iteration: int = 0
        self.subiter: int = 0
        self.iter_path: Path = self.out_path / str(self.iteration) / str(self.subiter)
        self.dispatcher: Dispatcher | None = None

    @abstractmethod
    def dispatch_fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        subiter: int,
    ) -> int:
        pass

    @abstractmethod
    def collect_loss(self, wait_id: int, iteration: int, subiter: int) -> float:
        pass

    @abstractmethod
    def _write_input_file(self, out_filepath: Path):
        pass

    @abstractmethod
    def _collect_raw_errors(self, errors_filepath: Path) -> pd.DataFrame:
        pass

    def _prep_fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        subiter: int,
    ):
        """
        Prepare hyperparameters for the model fitting.

        Parameters
        ----------
        opt_values : dict
            Dictionary of parameter names and values returned by the optimiser
            for the current iteration of fitting.
        iteration : int
            The current iteration number.
        subiter : int
            The current subiteration number.
        """
        self.iteration = iteration
        self.subiter = subiter
        self.iter_path = self.out_path / str(self.iteration) / str(self.subiter)
        self.iter_path.mkdir(parents=True, exist_ok=True)

        self.mlp_total = load.reconstitute_lists(self.mlp_total, opt_values)
        self.mlp_total = load.prep_dict_for_dump(self.mlp_total)
        self.mlp_total = load.trim_empty_values(self.mlp_total)  # type: ignore
        self.mlp_total = load.convert_numpy_types(self.mlp_total)

    def _validate_errors(
        self,
        errors: dict[str, list[float]],
        metric: Callable = maths.get_rmse,
        n_scaling: float | list[float] = 0.5,
    ) -> list[tuple[float, float]]:
        """
        Calculate the training and validation error values specific to the loss
        function of XPOT from the MLP.

        Parameters
        ----------
        errors : tuple
            Tuple of training and validation errors.
        metric : Callable
            The error metric to use. Default is RMSE, MAE is also available.
        """
        if isinstance(n_scaling, float):
            energy_diff = (
                errors[ColKW.ENERGY]
                / np.array(errors[ColKW.ATOMS]) ** n_scaling
            )
            forces_diff = errors[ColKW.FORCES]
            print(forces_diff)
            energy_error = metric(energy_diff)
            forces_error = metric(forces_diff)

            return [(energy_error, forces_error)]

        if isinstance(n_scaling, list):
            energy_diffs = []
            for i in n_scaling:
                energy_diff = (
                    errors[ColKW.ENERGY] / np.array(errors[ColKW.ATOMS]) ** i
                )
                energy_error = metric(energy_diff)
                energy_diffs.append(energy_error)
            forces_diff = errors[ColKW.FORCES]
            forces_error = metric(forces_diff)
            forces_diffs = [forces_error] * len(energy_diffs)
            output = []
            for e, f in zip(energy_diffs, forces_diffs):
                output.append((e, f))
            return output

        raise ValueError("n_scaling must be a float or list of floats.")

    def _process_errors(
        self,
        train_errors: tuple[float, float],
        test_errors: tuple[float, float],
        filepath: Path,
    ) -> float:
        """
        Write the error metrics to a file.

        Parameters
        ----------
        train_errors : tuple
            Tuple of training errors.
        test_errors : tuple
            Tuple of test errors.
        filename : str
            The file to write to.
        """
        errors = [
            train_errors[0],
            test_errors[0],
            train_errors[1],
            test_errors[1],
        ]
        self._write_error_file(errors, filepath)

        loss = maths.calculate_loss(test_errors[0], test_errors[1], self.energy_weight)
        return loss

    def _write_error_file(self,
                         errors: list[float],
                         filepath: Path):
        """
        Write the error values to a file.

        Parameters
        ----------
        e_train : float
            The energy training error.
        f_train : float
            The force training error.
        e_test : float
            The energy validation error.
        f_test : float
            The force validation error.
        filename : str
            The file to write to.
        """
        if len(errors) != 4:
            raise ValueError(
                "Error values must be a list of length 4, made up of "
                "the training and testing energy and force errors."
            )
        output_data = [self.iteration, self.subiter, *errors]
        with filepath.open("a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(output_data)
