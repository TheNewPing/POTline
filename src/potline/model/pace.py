"""
Pacemaker wrapper for fitting ACE potentials using XPOT in HPC.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from collections.abc import Callable

import yaml
import numpy as np
import pandas as pd
from xpot import maths # type: ignore

from .model import PotModel, RawLosses, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME, Losses, gen_from_template
from ..dispatcher import SupportedModel

LAST_POTENTIAL_NAME: str = 'output_potential.yaml'

class PotPACE(PotModel):
    """
    PACE implementation.
    """
    @staticmethod
    def get_fit_cmd(deep: bool = False) -> str:
        return  ' '.join(['pacemaker', CONFIG_NAME] + ([f'-p {LAST_POTENTIAL_NAME}'] if deep else []))

    def collect_loss(self) -> Losses:
        return self._validate_errors(self._calculate_errors())

    def lampify(self) -> Path:
        subprocess.run(['pace_yaml2yace', '-o',
                        str(self._yace_path),
                        str(self._out_path / LAST_POTENTIAL_NAME)],
                        check=True)
        return self._yace_path

    def create_potential(self) -> Path:
        potential_values: dict = {
            'pstyle': 'pace product',
            'yace_path': str(self._yace_path),
        }
        gen_from_template(POTENTIAL_TEMPLATE_PATH, potential_values, self._lmp_pot_path)
        return self._lmp_pot_path

    def set_config_maxiter(self, maxiter: int):
        with self._config_filepath.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        config['fit']['maxiter'] = maxiter

        with self._config_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

    def get_lammps_params(self) -> str:
        return '-k on g 1 -sf kk -pk kokkos newton on neigh half'

    def get_name(self) -> SupportedModel:
        return SupportedModel.PACE

    def _collect_raw_errors(self) -> pd.DataFrame:
        """
        Collect errors from the fitting process.

        Returns
            pd.DataFrame: the errors from the fitting process
        """
        errors_filepath: Path = self._out_path / "test_pred.pckl.gzip"
        df = pd.read_pickle(errors_filepath, compression="gzip")
        return df

    def _calculate_errors(self) -> RawLosses:
        """
        Validate the potential from pickle files produced by :code:`pacemaker`
        during the fitting process.

        Returns
            RawLosses: the raw losses for each prediction.
        """
        errors = self._collect_raw_errors()

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

    def _validate_errors(
        self,
        errors: RawLosses,
        metric: Callable = maths.get_rmse,
        n_scaling: float = 1,
    ) -> Losses:
        """
        Calculate the loss resulting loss.

        Args:
            - errors: the raw errors from the fitting process.
            - metric: the metric to use for the loss.
            - n_scaling: the scaling factor for the atom count.

        Returns:
            Losses: the losses from the fitting process.
        """
        energy_diff = (
            errors.energies
            / np.array(errors.atom_counts) ** n_scaling
        )
        return Losses(metric(energy_diff), metric(errors.forces))

    @staticmethod
    def from_path(out_path):
        return PotPACE(out_path / CONFIG_NAME, out_path)
