"""
Pacemaker wrapper for fitting ACE potentials using XPOT in HPC.
"""

from __future__ import annotations

import sys
import shutil
import json
from pathlib import Path

import yaml
from mace.cli.create_lammps_model import main as create_lammps_model

from .model import PotModel, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME, Losses, gen_from_template
from ..dispatcher import SupportedModel

LAST_POTENTIAL_NAME: str = 'output_potential.yaml'

class PotMACE(PotModel):
    """
    MACE implementation.
    """
    @staticmethod
    def get_fit_cmd(deep: bool = False,):
        return ' '.join(['mace_run_train', f'--config {CONFIG_NAME}'] +
                     (['--restart_latest'] if deep else []))

    def collect_loss(self) -> Losses:
        results_path: Path = next((self._out_path / "results").glob("*.txt"))
        with results_path.open('r', encoding='utf-8') as file:
            lines = file.readlines()

        eval_lines: list[str] = [line for line in lines if '"mode": "eval"' in line]
        if not eval_lines:
            raise ValueError("No evaluation data found.")

        last_eval = eval_lines[-1]
        last_eval_data: dict = json.loads(last_eval)

        rmse_e: float = float(last_eval_data["rmse_e"])
        rmse_f: float = float(last_eval_data["rmse_f"])

        return Losses(rmse_e, rmse_f)

    def lampify(self) -> Path:
        with self._config_filepath.open('r', encoding='utf-8') as file:
            model_name: str = yaml.safe_load(file)['name']

        # if the model is trained with swa, names are different
        if (self._out_path / (model_name + '_stagetwo.model')).exists():
            model_filepath = self._out_path / (model_name + '_stagetwo.model')
            self._yace_path = self._out_path / f'{model_name}_stagetwo.model-lammps.pt'
        else:
            model_filepath: Path = self._out_path / (model_name + '.model')
            self._yace_path = self._out_path / f'{model_name}.model-lammps.pt'

        old_argv = sys.argv
        sys.argv = ["program", str(model_filepath)]
        create_lammps_model()
        sys.argv = old_argv

        return self._yace_path

    def create_potential(self) -> Path:
        potential_values: dict = {
            'pstyle': 'mace no_domain_decomposition',
            'yace_path': str(self._yace_path),
        }
        gen_from_template(POTENTIAL_TEMPLATE_PATH, potential_values, self._lmp_pot_path)
        return self._lmp_pot_path

    def set_config_maxiter(self, maxiter: int):
        with self._config_filepath.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        config['max_num_epochs'] = maxiter

        with self._config_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

    @staticmethod
    def get_lammps_params() -> str:
        return ''

    def get_name(self) -> SupportedModel:
        return SupportedModel.MACE

    def switch_out_path(self, out_path: Path):
        shutil.copytree(self._out_path / 'checkpoints', out_path / 'checkpoints', dirs_exist_ok=True)
        super().switch_out_path(out_path)
