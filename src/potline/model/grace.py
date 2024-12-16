"""
Pacemaker wrapper for fitting ACE potentials using XPOT in HPC.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import yaml

from .model import PotModel, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME, Losses
from ..dispatcher import DispatcherFactory, SupportedModel
from ..utils import gen_from_template

LAST_POTENTIAL_NAME: str = 'output_potential.yaml'

class PotGRACE(PotModel):
    """
    GRACE implementation.
    Requires gracemaker.
    """
    def __init__(self, config_filepath, out_path):
        super().__init__(config_filepath, out_path)
        with config_filepath.open('r', encoding='utf-8') as file:
            config: dict = yaml.safe_load(file)
            self._seed_number: int = config['seed']
            self._seed_path: Path = out_path / 'seed' / f'{self._seed_number}'
        self._yace_path = self._seed_path / 'final_model'

    def dispatch_fit(self,
                     dispatcher_factory: DispatcherFactory,
                     deep: bool = False):
        commands: list[str] = [
            f'cd {self._out_path}',
            ' '.join(['gracemaker', str(self._config_filepath)] +
                     (['-r'] if deep else []))
        ]
        self._dispatcher = dispatcher_factory.create_dispatcher(
            commands, self._out_path, SupportedModel.GRACE.value)
        self._dispatcher.dispatch()

    def collect_loss(self) -> Losses:
        if self._dispatcher is None:
            raise ValueError("Dispatcher not set.")
        self._dispatcher.wait()
        train_metrics_path: Path = self._seed_path / 'train_metrics.yaml'
        with train_metrics_path.open('r', encoding='utf-8') as file:
            train_metrics: dict = yaml.safe_load(file)

        rmse_de: float = train_metrics[-1]['rmse/de']
        rmse_f_comp: float = train_metrics[-1]['rmse/f_comp']

        return Losses(rmse_de, rmse_f_comp)

    def lampify(self) -> Path:
        """
        Convert the model YAML to YACE format.

        Returns:
            Path: The path to the YACE file.
        """
        return self._yace_path

    def create_potential(self) -> Path:
        """
        Create the potential in YACE format.

        Returns:
            Path: The path to the potential.
        """
        potential_values: dict = {
            'pstyle': 'grace pad_verbose',
            'yace_path': str(self._yace_path),
        }
        gen_from_template(POTENTIAL_TEMPLATE_PATH, potential_values, self._lmp_pot_path)
        return self._lmp_pot_path

    def set_config_maxiter(self, maxiter: int):
        """
        Set the maximum number of iterations in the configuration file.
        """
        with self._config_filepath.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        config['fit']['maxiter'] = maxiter

        with self._config_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

    def get_lammps_params(self) -> str:
        """
        Get the LAMMPS parameters.
        """
        return ''

    def switch_out_path(self, out_path: Path):
        """
        Switch the output path of the model.
        """
        shutil.copytree(self._out_path, out_path, dirs_exist_ok=True)
        super().switch_out_path(out_path)
        self._seed_path: Path = self._out_path / 'seed' / f'{self._seed_number}'
        self._yace_path = self._seed_path / 'final_model'

    @staticmethod
    def from_path(out_path):
        """
        Create a model from a path.
        """
        return PotGRACE(out_path / CONFIG_NAME, out_path)
