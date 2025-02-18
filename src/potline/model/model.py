"""
Base class for MLPs in XPOT using HPC.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from string import Template

import yaml
import numpy as np

YACE_NAME: str = 'model.yace'
POTENTIAL_NAME: str = 'potential.in'
CONFIG_NAME: str = "optimized_params.yaml"
POTENTIAL_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / POTENTIAL_NAME

class Losses():
    """
    Losses class for the model.

    Args:
        - energy: energy loss
        - force: force loss
    """
    def __init__(self, energy: float, force: float):
        self.energy: float = energy if not math.isnan(energy) else float(np.finfo(np.float32).max)
        self.force: float = force if not math.isnan(force) else float(np.finfo(np.float32).max)

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

def gen_from_template(template_path: Path, values: dict[str, str | int | float | Path], out_filepath: Path):
    """
    Generate a file from a template file.
    """
    with template_path.open('r', encoding='utf-8') as file_template:
        template: Template = Template(file_template.read())
        content: str = template.safe_substitute(values)
        with out_filepath.open('w', encoding='utf-8') as file_out:
            file_out.write(content)

class PotModel(ABC):
    """
    Base class for MLIAP models.

    Args:
        - out_path: path to the output directory.
        - pretrained: flag for pretrained model.
    """
    def __init__(self, out_path: Path, pretrained: bool = False):
        self._out_path: Path = out_path
        self._config_filepath: Path = self._out_path / CONFIG_NAME
        self._yace_path: Path = self._out_path / YACE_NAME
        self._lmp_pot_path: Path = self._out_path / POTENTIAL_NAME

    @staticmethod
    @abstractmethod
    def get_fit_cmd(deep: bool = False,) -> str:
        """
        Dispatch the fitting process.

        Args:
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

    @staticmethod
    @abstractmethod
    def get_lammps_params() -> str:
        """
        Get model specific LAMMPS parameters.

        Returns:
            str: the model specific LAMMPS parameters.
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
        self._out_path = out_path
        self._config_filepath = self._out_path / self._config_filepath.name
        self._yace_path = self._out_path / YACE_NAME
        self._lmp_pot_path = self._out_path / POTENTIAL_NAME

    def get_params(self) -> dict:
        """
        Get the parameters of the model.

        Returns:
            dict: the parameters of the model.
        """
        with self._config_filepath.open('r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def get_pot_path(self) -> Path:
        """
        Get the path to the potential file.
        """
        return self._lmp_pot_path
