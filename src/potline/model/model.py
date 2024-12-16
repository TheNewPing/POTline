"""
Base class for MLPs in XPOT using HPC.
"""

from __future__ import annotations

import shutil
import json
from pathlib import Path
from abc import ABC, abstractmethod

import yaml

from ..dispatcher import Dispatcher, SupportedModel, DispatcherFactory

YACE_NAME: str = 'model.yace'
POTENTIAL_NAME: str = 'potential.in'
CONFIG_NAME: str = "optimized_params.yaml"
POTENTIAL_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / POTENTIAL_NAME

# TODO: move to model implementation
_MODEL_DEFAULTS = {
    SupportedModel.PACE: Path(__file__).parent / "defaults" / "ace_defaults.json",
    SupportedModel.MACE: Path(__file__).parent / "defaults" / "mace_defaults.json",
    SupportedModel.GRACE: Path(__file__).parent / "defaults" / "grace_defaults.json",
}

class Losses():
    """
    Losses class for the model.

    Args:
        - energy: energy loss
        - force: force loss
    """
    def __init__(self, energy: float, force: float):
        self.energy: float = energy
        self.force: float = force

class RawLosses():
    """
    Raw losses class for the model.

    Args:
        - energies: list of energy losses
        - forces: list of force losses
        - atom_counts: list of atom counts
    """
    def __init__(self, energies: list[float], forces: list[float],
                 atom_counts: list[float]):
        self.energies: list[float] = energies
        self.forces: list[float] = forces
        self.atom_counts: list[float] = atom_counts

class PotModel(ABC):
    """
    Base class for MLIAP models.

    Args:
        - config_filepath: path to the configuration file.
        - out_path: path to the output directory.
    """
    def __init__(self, config_filepath: Path,
                 out_path: Path):
        self._config_filepath: Path = config_filepath
        self._out_path: Path = out_path
        self._dispatcher: Dispatcher | None = None
        self._yace_path: Path = self._out_path.parent / YACE_NAME
        self._lmp_pot_path: Path = self._out_path.parent / POTENTIAL_NAME

    @abstractmethod
    def dispatch_fit(self,
                     dispatcher_factory: DispatcherFactory,
                     deep: bool = False,):
        """
        Dispatch the fitting process.

        Args:
            - dispatcher_factory: factory for dispatching the fitting process.
            - deep: flag for deep training.
        """

    @abstractmethod
    def collect_loss(self) -> Losses:
        """
        Collect the loss from the fitting process.

        Returns:
            Losses: the losses from the fitting process.
        """

    @abstractmethod
    def lampify(self) -> Path:
        """
        Convert the model to a LAMMPS compatible format.

        Returns:
            Path: The path to the converted model.
        """

    @abstractmethod
    def create_potential(self) -> Path:
        """
        Create the potential file that will be included in the LAMMPS scripts.

        Returns:
            Path: The path to the potential.
        """

    @abstractmethod
    def set_config_maxiter(self, maxiter: int):
        """
        Set the maximum number of iterations for the model.
        Overwrites the current configuration file. Used for deep training.

        Args:
            - maxiter: the maximum number of iterations.
        """

    @abstractmethod
    def get_lammps_params(self) -> str:
        """
        Get model specific LAMMPS parameters.

        Returns:
            str: the model specific LAMMPS parameters.
        """

    @abstractmethod
    def get_name(self) -> SupportedModel:
        """
        Get the name of the model.

        Returns:
            SupportedModel: the name of the model.
        """

    def get_out_path(self) -> Path:
        """
        Get the output path of the model.

        Returns:
            Path: the output path of the model.
        """
        return self._out_path

    def switch_out_path(self, out_path: Path):
        """
        Switch the output path of the model.

        Args:
            - out_path: the new output path.
        """
        shutil.copy(self._config_filepath, out_path)
        self._config_filepath = out_path / self._config_filepath.name
        self._out_path = out_path

    def get_params(self) -> dict:
        """
        Get the parameters of the model.

        Returns:
            dict: the parameters of the model.
        """
        with self._config_filepath.open('r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    @staticmethod
    def get_defaults(model_name: str) -> dict:
        """
        Get the default parameters of the model class.

        Args:
            - model_name: name of the model

        Returns:
            dict: the default parameters of the model.
        """
        for model in SupportedModel:
            if model.value == model_name:
                with _MODEL_DEFAULTS[model].open('r', encoding='utf-8') as file:
                    return json.load(file)

        raise ValueError(f"Model {model_name} not supported.")

    @staticmethod
    @abstractmethod
    def from_path(out_path: Path) -> PotModel:
        """
        Create a model from a path. The model must be already trained.

        Args:
            - out_path: path to the model.

        Returns:
            PotModel: the model.
        """
