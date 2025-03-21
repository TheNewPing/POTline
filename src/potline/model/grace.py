"""
Gracemaker wrapper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import shutil

import yaml

from .model import PotModel, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME, Losses, gen_from_template
from ..dispatcher import SupportedModel

LAST_POTENTIAL_NAME: str = 'output_potential.yaml'

class PotGRACE(PotModel):
    """
    GRACE implementation.
    """
    def __init__(self, out_path: Path, pretrained: bool = False):
        super().__init__(out_path, pretrained)
        if self._pretrained:
            self._yace_path = out_path
        else:
            with self._config_filepath.open('r', encoding='utf-8') as file:
                config: dict = yaml.safe_load(file)
                self._seed_number: int = config['seed']
                self._seed_path: Path = out_path / 'seed' / f'{self._seed_number}'
                pot: dict = config['potential']
                self._preset: str = pot.get('preset', 'default')
            if self._preset == 'FS':
                self._yace_path = self._seed_path / 'FS_model.yaml'
            else:
                self._yace_path = self._seed_path / 'saved_model'

    @staticmethod
    def get_fit_cmd(deep: bool = False):
        return ' '.join(['gracemaker', CONFIG_NAME] + (['-r'] if deep else []))

    def collect_loss(self) -> Losses:
        if self._pretrained:
            raise NotImplementedError('Pretrained model does not support loss collection.')

        train_metrics_path: Path = self._seed_path / 'train_metrics.yaml'
        with train_metrics_path.open('r', encoding='utf-8') as file:
            train_metrics: dict = yaml.safe_load(file)

        rmse_de: float = float(train_metrics[-1]['rmse/depa'])
        rmse_f_comp: float = float(train_metrics[-1]['rmse/f_comp'])

        return Losses(rmse_de, rmse_f_comp)

    def lampify(self) -> Path:
        if not self._pretrained:
            cmd: list[str] = ['gracemaker', '-r', '-s']
            if self._preset == 'FS':
                cmd.append('-sf')
            cmd.append(CONFIG_NAME)
            subprocess.run(cmd, check=True, cwd=self._out_path)

        return self._yace_path

    def create_potential(self) -> Path:
        preset: str = 'grace/fs' \
            if not self._pretrained and self._preset == 'FS' \
            else 'grace pad_verbose'
        potential_values: dict = {
            'pstyle': preset,
            'yace_path': str(self._yace_path),
        }
        gen_from_template(POTENTIAL_TEMPLATE_PATH, potential_values, self._lmp_pot_path)
        return self._lmp_pot_path

    def set_config_maxiter(self, maxiter: int):
        if self._pretrained:
            raise NotImplementedError('Pretrained model does not support maxiter setting.')

        with self._config_filepath.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        config['fit']['maxiter'] = maxiter

        with self._config_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

    @staticmethod
    def get_lammps_params() -> str:
        return ''

    def get_name(self) -> SupportedModel:
        return SupportedModel.GRACE

    def switch_out_path(self, out_path: Path):
        if self._pretrained:
            raise NotImplementedError('Pretrained model does not support out path switching.')

        shutil.copytree(self._out_path, out_path, dirs_exist_ok=True)
        super().switch_out_path(out_path)
        self._seed_path = self._out_path / 'seed' / f'{self._seed_number}'
        if self._preset == 'FS':
            self._yace_path = self._seed_path / 'FS_model.yaml'
        else:
            self._yace_path = self._seed_path / 'final_model'
