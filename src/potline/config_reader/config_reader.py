"""
Configuration file reader for the optimization pipeline.
"""
from pathlib import Path

import hjson # type: ignore

from ..utils import patify
from ..optimizer import Optimizer
from ..optimizer.xpot_adapter import XpotAdapter

class ConfigReader():
    """
    Class for reading and converting configuration files.
    A config file is written in hjson format and should have the following main sections:
    - optimizer: contains the configuration for the optimizer, uses the XPOT format.
    - inference: contains the configuration for the inference benchmark with LAMMPS.
    - data_analysis: contains the configuration for the data analysis on mechanical properties with LAMMPS.
    - lammps: contains the configuration for the LAMMPS simulation.
    """
    def __init__(self, file_path: Path):
        self.file_path: Path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            self.config_data: dict = hjson.load(file)

    def create_optimizer(self) -> Optimizer:
        """
        Convert the configuration file to the XPOT format and create an optimizer.
        """
        converted_file_path: Path = self.file_path.with_stem(self.file_path.stem + '_converted')
        if 'optimizer' not in self.config_data:
            raise ValueError('No optimizer configuration found in the config file.')
        if 'xpot' not in self.config_data['optimizer']:
            raise ValueError('No XPOT configuration found in the optimizer configuration.')

        # add model name
        self.config_data['optimizer']['xpot']['fitting_executable'] = \
            self.config_data['general']['model_name']

        with open(converted_file_path, 'w', encoding='utf-8') as converted_file:
            hjson.dump(self.config_data['optimizer'], converted_file)
        return XpotAdapter(converted_file_path)

    def get_inf_benchmark_config(self) -> dict[str, str | int | float | Path]:
        if 'inference' not in self.config_data:
            raise ValueError('No inference configuration found in the config file.')
        return patify(self.config_data['inference'])

    def get_data_analysis_config(self) -> dict[str, str | int | float | Path]:
        if 'data_analysis' not in self.config_data:
            raise ValueError('No data analysis configuration found in the config file.')
        return patify(self.config_data['data_analysis'])

    def get_lammps_bin_path(self) -> Path:
        if 'general' not in self.config_data or 'lammps_bin_path' not in self.config_data['general']:
            raise ValueError('No LAMMPS binary path found in the config file.')
        return Path(self.config_data['general']['lammps_bin_path'])

    def get_out_yace_path(self) -> Path:
        if 'general' not in self.config_data or 'out_yace_path' not in self.config_data['general']:
            raise ValueError('No YACE output path found in the config file.')
        return Path(self.config_data['general']['out_yace_path'])

    def get_model_name(self) -> str:
        if 'general' not in self.config_data or 'model_name' not in self.config_data['general']:
            raise ValueError('No model name found in the config file.')
        return self.config_data['general']['model_name']

    def get_best_n_models(self) -> int:
        if 'general' not in self.config_data or 'best_n_models' not in self.config_data['general']:
            raise ValueError('No best n models found in the config file.')
        return self.config_data['general']['best_n_models']
