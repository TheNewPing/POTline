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

from .model import PotModel, POTENTIAL_TEMPLATE_PATH, CONFIG_NAME, Losses
from ..dispatcher import DispatcherFactory, SupportedModel
from ..utils import gen_from_template

LAST_POTENTIAL_NAME: str = 'output_potential.yaml'

class PotMACE(PotModel):
    """
    MACE implementation.
    """
    def dispatch_fit(self,
                     dispatcher_factory: DispatcherFactory,
                     deep: bool = False,):
        commands: list[str] = [
            f'cd {self._out_path}',
            ' '.join(['mace_run_train', f'--config {str(self._config_filepath)}'] +
                     (['--restart_latest'] if deep else []))
        ]
        self._dispatcher = dispatcher_factory.create_dispatcher(
            commands, self._out_path, SupportedModel.MACE.value)
        self._dispatcher.dispatch()

    def collect_loss(self) -> Losses:
        if self._dispatcher is None:
            raise ValueError("Dispatcher not set.")
        self._dispatcher.wait()
        results_path: Path = next((self._out_path / "results").glob("*.txt"))
        with results_path.open('r', encoding='utf-8') as file:
            lines = file.readlines()

        eval_lines: list[str] = [line for line in lines if '"mode": "eval"' in line]
        if not eval_lines:
            raise ValueError("No evaluation data found.")

        last_eval = eval_lines[-1]
        last_eval_data: dict = json.loads(last_eval)

        rmse_e: float | None = last_eval_data.get("rmse_e")
        rmse_f: float | None = last_eval_data.get("rmse_f")

        if rmse_e is None or rmse_f is None:
            raise ValueError("RMSE values not found in the last evaluation data.")

        return Losses(rmse_e, rmse_f)

    def lampify(self) -> Path:
        """
        Convert the model YAML to YACE format.

        Returns:
            Path: The path to the YACE file.
        """
        # TODO: insert path in config
        with self._config_filepath.open('r', encoding='utf-8') as file:
            model_name: str = yaml.safe_load(file)['name']

        old_argv = sys.argv
        sys.argv = ["program", f'{self._out_path / model_name}.model']
        create_lammps_model()
        sys.argv = old_argv

        self._yace_path = self._out_path / f'{model_name}.model-lammps.pt'
        return self._yace_path

    def create_potential(self) -> Path:
        """
        Create the potential in YACE format.

        Returns:
            Path: The path to the potential.
        """
        potential_values: dict = {
            'pstyle': 'mace no_domain_decomposition',
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

        config['max_num_epochs'] = maxiter

        with self._config_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

    def get_lammps_params(self) -> str:
        """
        Get the LAMMPS parameters.
        """
        return '-k on g 1 -sf kk -pk kokkos newton on neigh half'

    def get_name(self) -> SupportedModel:
        return SupportedModel.MACE

    def switch_out_path(self, out_path: Path):
        """
        Switch the output path of the model.
        """
        shutil.copytree(self._out_path / 'checkpoints', out_path / 'checkpoints', dirs_exist_ok=True)
        super().switch_out_path(out_path)

    @staticmethod
    def from_path(out_path):
        """
        Create a model from a path.
        """
        return PotMACE(out_path / CONFIG_NAME, out_path)
