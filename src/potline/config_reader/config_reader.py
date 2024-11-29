"""
Configuration file reader for the optimization pipeline.
"""
from pathlib import Path

import hjson # type: ignore

from ..utils import patify
from ..optimizer import Optimizer
from ..optimizer.xpot_adapter import XpotAdapter

GEN_NAME: str = 'general'
LMP_BIN_NAME: str = 'lammps_bin_path'
MODEL_NAME: str = 'model_name'
BEST_N_NAME: str = 'best_n_models'

DT_NAME: str = 'deep_train'

INF_NAME: str = 'inference'

DA_NAME: str = 'data_analysis'

HYP_NAME: str = 'hyper_search'
HYP_ITER_NAME: str = 'hyper_search_iterations'
GRID_NAME: str = 'initial_grid'
FUNC_NAME: str = 'func_levels'


ConfigDict = dict[str, str | int | float | Path]

class ConfigReader():
    """
    Class for reading and converting configuration files.
    A config file is written in hjson format and should have the following main sections:
    - hyper_search: contains the configuration for the optimizer, uses the XPOT format.
    - deep_train: contains the configuration for the deep training after the hyperparameter search.
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
        if 'hyper_search' not in self.config_data:
            raise ValueError('No optimizer configuration found in the config file.')
        if 'xpot' not in self.config_data['hyper_search']:
            raise ValueError('No XPOT configuration found in the optimizer configuration.')

        # add model name
        self.config_data['hyper_search']['xpot']['fitting_executable'] = \
            self.config_data['general']['model_name']

        with open(converted_file_path, 'w', encoding='utf-8') as converted_file:
            hjson.dump(self.config_data['hyper_search'], converted_file)
        return XpotAdapter(converted_file_path)

    def get_config_section(self, section_name: str) -> ConfigDict:
        if section_name not in self.config_data:
            raise ValueError(f'No {section_name} configuration found in the config file.')
        return patify(self.config_data[section_name])
