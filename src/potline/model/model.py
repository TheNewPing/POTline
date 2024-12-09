"""
Base class for MLPs in XPOT using HPC.
"""

from __future__ import annotations

import shutil
import json
from collections.abc import Callable
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from xpot import maths # type: ignore

from ..dispatcher import Dispatcher, SupportedModel, DispatcherFactory

YACE_NAME: str = 'model.yace'
POTENTIAL_NAME: str = 'potential.in'
POTENTIAL_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / POTENTIAL_NAME

class Losses():
    """
    Losses class for the model.
    """
    def __init__(self, energy: float, force: float):
        self.energy: float = energy
        self.force: float = force

class RawLosses():
    """
    Raw losses class for the model.
    """
    def __init__(self, energies: list[float], forces: list[float],
                 atom_counts: list[float]):
        self.energies: list[float] = energies
        self.forces: list[float] = forces
        self.atom_counts: list[float] = atom_counts

class ModelTracker():
    """
    Class to track the progress of a job in the optimisation sweep.
    """
    def __init__(self, model: PotModel, iteration: int, subiter: int,
                 params: dict, train_losses: Losses | None = None,
                 test_losses: Losses | None = None) -> None:
        self.model = model
        self.iteration = iteration
        self.subiter = subiter
        self.params = params
        self.train_losses = train_losses
        self.test_losses = test_losses

    def get_total_test_loss(self, energy_weight: float) -> float:
        """
        Get the total test loss from the model.
        """
        if self.test_losses is None:
            raise ValueError("Test loss not calculated.")
        return maths.calculate_loss(self.test_losses.energy, self.test_losses.force, energy_weight)

    def get_total_train_loss(self, energy_weight: float) -> float:
        """
        Get the total train loss from the model.
        """
        if self.train_losses is None:
            raise ValueError("Train loss not calculated.")
        return maths.calculate_loss(self.train_losses.energy, self.train_losses.force, energy_weight)

_MODEL_DEFAULTS = {
    SupportedModel.PACE: Path(__file__).parent / "defaults" / "ace_defaults.json",
    SupportedModel.MACE: Path(__file__).parent / "defaults" / "mace_defaults.json",
}

class PotModel(ABC):
    """
    Base class for MLIAP models.

    Args:
        - config_filepath: path to the configuration file.
        - out_path: path to the output directory.
        - hpc: flag to run the simulations on HPC.
    """
    def __init__(self, config_filepath: Path,
                 out_path: Path,
                 hpc: bool):
        self._config_filepath: Path = config_filepath
        self._hpc: bool = hpc
        self._out_path: Path = out_path
        self._dispatcher: Dispatcher | None = None
        self._yace_path: Path = self._out_path.parent / YACE_NAME
        self._lmp_pot_path: Path = self._out_path.parent / POTENTIAL_NAME

    @abstractmethod
    def dispatch_fit(self,
                     dispatcher_factory: DispatcherFactory,
                     extra_cmd_opts: list[str] | None = None):
        pass

    @abstractmethod
    def set_config_maxiter(self, maxiter: int):
        pass

    @abstractmethod
    def lampify(self) -> Path:
        """
        Convert the model YAML to YACE format.

        Returns:
            Path: The path to the YACE file.
        """

    @abstractmethod
    def create_potential(self) -> Path:
        """
        Create the potential in YACE format.

        Returns:
            Path: The path to the potential.
        """

    @abstractmethod
    def get_last_pot_path(self) -> Path:
        pass

    @abstractmethod
    def _collect_raw_errors(self, validation: bool) -> pd.DataFrame:
        pass

    @abstractmethod
    def _calculate_errors(self, validation: bool = False) -> RawLosses:
        pass

    def collect_loss(self, validation: bool) -> Losses:
        """
        Collect the loss from the fitting process.
        """
        if self._dispatcher is None:
            raise ValueError("Dispatcher not set.")
        self._dispatcher.wait()
        return self._validate_errors(self._calculate_errors(validation))

    def _validate_errors(
        self,
        errors: RawLosses,
        metric: Callable = maths.get_rmse,
        n_scaling: float = 1,
    ) -> Losses:
        """
        Calculate the training and validation error values specific to the loss
        function of XPOT from the MLP.
        """
        energy_diff = (
            errors.energies
            / np.array(errors.atom_counts) ** n_scaling
        )
        return Losses(metric(energy_diff), metric(errors.forces))

    def get_out_path(self) -> Path:
        """
        Get the output path of the model.
        """
        return self._out_path

    def switch_out_path(self, out_path: Path):
        """
        Switch the output path of the model.
        """
        shutil.copy(self._config_filepath, out_path)
        self._config_filepath = out_path / self._config_filepath.name
        self._out_path = out_path

    @staticmethod
    def get_defaults(model_name: str) -> dict:
        """
        Get the default parameters from a json file.
        """
        for model in SupportedModel:
            if model.value == model_name:
                with _MODEL_DEFAULTS[model].open('r', encoding='utf-8') as file:
                    return json.load(file)

        raise ValueError(f"Model {model_name} not supported.")
