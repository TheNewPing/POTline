"""
Pacemaker wrapper for fitting ACE potentials using XPOT in HPC.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import shutil

import yaml
import pandas as pd

from .model import PotModel, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME, Losses, gen_from_template
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
        test_metrics_path: Path = self._out_path / 'test_metrics.txt'
        with test_metrics_path.open('r', encoding='utf-8') as file:
            train_metrics = pd.read_csv(file, delim_whitespace=True).to_dict(orient='records')

        rmse_de: float = float(train_metrics[-1]['rmse_epa'])
        rmse_f_comp: float = float(train_metrics[-1]['rmse_f_comp'])

        return Losses(rmse_de, rmse_f_comp)

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

    @staticmethod
    def get_lammps_params() -> str:
        return '-k on g 1 -sf kk -pk kokkos newton on neigh half'

    def get_name(self) -> SupportedModel:
        return SupportedModel.PACE

    def switch_out_path(self, out_path: Path):
        shutil.copy(self._out_path / LAST_POTENTIAL_NAME, out_path / LAST_POTENTIAL_NAME)
        super().switch_out_path(out_path)

    def _collect_raw_errors(self) -> pd.DataFrame:
        """
        Collect errors from the fitting process.

        Returns
            pd.DataFrame: the errors from the fitting process
        """
        errors_filepath: Path = self._out_path / "test_pred.pckl.gzip"
        df = pd.read_pickle(errors_filepath, compression="gzip")
        return df
