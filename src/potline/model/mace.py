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
    def __init__(self, out_path: Path, pretrained: bool = False):
        super().__init__(out_path, pretrained)
        if self._pretrained:
            self._model_name: str = self._out_path.name
            self._out_path = self._out_path.parent
            self._lmp_pot_path = self._out_path / self._lmp_pot_path.name
        else:
            with self._config_filepath.open('r', encoding='utf-8') as file:
                self._model_name = yaml.safe_load(file)['name']
    @staticmethod
    def get_fit_cmd(deep: bool = False,):
        return ' '.join(['mace_run_train', f'--config {CONFIG_NAME}'] +
                     (['--restart_latest'] if deep else []))

    def collect_loss(self) -> Losses:
        if self._pretrained:
            raise NotImplementedError('Pretrained model does not support loss collection.')

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
        # if the model is trained with swa, names are different
        if (self._out_path / (self._model_name + '_stagetwo.model')).exists():
            model_filepath: Path = self._out_path / (self._model_name + '_stagetwo.model')
            self._yace_path = self._out_path / f'{self._model_name}_stagetwo.model-lammps.pt'
        elif self._pretrained:
            model_filepath = self._out_path / self._model_name
            self._yace_path = self._out_path / f'{self._model_name}.model-lammps.pt'
        else:
            model_filepath = self._out_path / (self._model_name + '.model')
            self._yace_path = self._out_path / f'{self._model_name}.model-lammps.pt'

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
        if self._pretrained:
            raise NotImplementedError('Pretrained model does not support maxiter setting.')

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
        if self._pretrained:
            raise NotImplementedError('Pretrained model does not support switching output path.')

        shutil.copytree(self._out_path / 'checkpoints', out_path / 'checkpoints', dirs_exist_ok=True)
        super().switch_out_path(out_path)
