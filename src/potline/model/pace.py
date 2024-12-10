"""
Pacemaker wrapper for fitting ACE potentials using XPOT in HPC.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from .model import PotModel, RawLosses, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME
from ..dispatcher import DispatcherFactory, SupportedModel
from ..utils import gen_from_template

LAST_POTENTIAL_NAME: str = 'output_potential.yaml'

class PotPACE(PotModel):
    """
    PACE implementation.
    Requires pacemaker.
    """
    def dispatch_fit(self,
                     dispatcher_factory: DispatcherFactory,
                     extra_cmd_opts: list[str] | None = None):
        commands: list[str] = [
            f'cd {self._out_path}',
            ' '.join(['pacemaker', str(self._config_filepath)] + (extra_cmd_opts or []))
        ]
        self._dispatcher = dispatcher_factory.create_dispatcher(commands, self._out_path, SupportedModel.PACE)
        self._dispatcher.dispatch()

    def set_config_maxiter(self, maxiter: int):
        """
        Set the maximum number of iterations in the configuration file.
        """
        with self._config_filepath.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        config['fit']['maxiter'] = maxiter

        with self._config_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

    def lampify(self) -> Path:
        """
        Convert the model YAML to YACE format.

        Returns:
            Path: The path to the YACE file.
        """
        subprocess.run(['pace_yaml2yace', '-o',
                        str(self._yace_path),
                        str(self._out_path / LAST_POTENTIAL_NAME)],
                        check=True)
        return self._yace_path

    def create_potential(self) -> Path:
        """
        Create the potential in YACE format.

        Returns:
            Path: The path to the potential.
        """
        potential_values: dict = {
            'pstyle': 'pace',
            'yace_path': str(self._yace_path),
        }
        gen_from_template(POTENTIAL_TEMPLATE_PATH, potential_values, self._lmp_pot_path)
        return self._lmp_pot_path

    def get_last_pot_path(self) -> Path:
        return self._out_path / LAST_POTENTIAL_NAME

    def _collect_raw_errors(self, validation: bool) -> pd.DataFrame:
        """
        Collect errors from the fitting process.
        """
        errors_filepath: Path = self._out_path / "test_pred.pckl.gzip" if validation \
            else self._out_path / "train_pred.pckl.gzip"
        df = pd.read_pickle(errors_filepath, compression="gzip")
        return df

    def _calculate_errors(self, validation: bool = False) -> RawLosses:
        """
        Validate the potential from pickle files produced by :code:`pacemaker`
        during the fitting process.

        Parameters
        ----------
        validation : bool
            If True, calculate validation errors, otherwise calculate training
            errors.

        Returns
        -------
        dict
            The errors as a dictionary of lists.
        """
        errors = self._collect_raw_errors(validation)

        n_per_structure = errors["NUMBER_OF_ATOMS"].values.tolist()

        ref_energy = errors["energy_corrected"].values.tolist()
        pred_energy = errors["energy_pred"].values.tolist()

        energy_diff = [pred - ref for pred, ref in zip(pred_energy, ref_energy)]

        ref_forces = np.concatenate(errors["forces"].to_numpy(), axis=None)
        pred_forces = np.concatenate(
            errors["forces_pred"].to_numpy(), axis=None
        )
        forces_diff = [pred - ref for pred, ref in zip(pred_forces, ref_forces)]

        return RawLosses(energy_diff, forces_diff, n_per_structure)

    @staticmethod
    def from_path(out_path):
        """
        Create a model from a path.
        """
        return PotPACE(out_path / CONFIG_NAME, out_path)
