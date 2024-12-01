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
PRE_STEPS_NAME: str = 'prerun_steps'
MAX_STEPS_NAME: str = 'max_steps'
N_CPU_NAME: str = 'n_cpu'

DA_NAME: str = 'data_analysis'
LAMMPS_INPS_NAME: str = 'lammps_inps_path'
PPS_PYTHON_NAME: str = 'pps_python_path'
REF_DATA_NAME: str = 'ref_data_path'

HYP_NAME: str = 'hyper_search'
MAX_ITER_NAME: str = 'max_iter'
N_INIT_PTS_NAME: str = 'n_initial_points'
N_PTS_NAME: str = 'n_points'
STRAT_NAME: str = 'strategy'
GRID_NAME: str = 'initial_grid'
FUNC_NAME: str = 'func_levels'

ConfigDict = dict[str, str | int | float | Path]

class BenchConfig():
    """
    Configuration class for the benchmarking step.
    """
    def __init__(self, lammps_bin_path: Path,
                 prerun_steps: int,
                 max_steps: int,
                 n_cpu: int):
        self.lammps_bin_path: Path = lammps_bin_path
        self.prerun_steps: int = prerun_steps
        self.max_steps: int = max_steps
        self.n_cpu: int = n_cpu

class PropConfig():
    """
    Configuration class for the data analysis step.
    """
    def __init__(self, lammps_bin_path: Path,
                 lammps_inps_path: Path,
                 pps_python_path: Path,
                 ref_data_path: Path):
        self.lammps_bin_path: Path = lammps_bin_path
        self.lammps_inps_path: Path = lammps_inps_path
        self.pps_python_path: Path = pps_python_path
        self.ref_data_path: Path = ref_data_path

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

        hyper_config: dict = self.config_data['hyper_search']
        # add model name
        hyper_config['xpot']['fitting_executable'] = \
            self.config_data['general']['model_name']

        kwargs = {
            N_INIT_PTS_NAME: hyper_config.pop(N_INIT_PTS_NAME),
            N_PTS_NAME: hyper_config.pop(N_PTS_NAME),
            STRAT_NAME: hyper_config.pop(STRAT_NAME),
        }
        max_iter: int = hyper_config.pop(MAX_ITER_NAME)

        with open(converted_file_path, 'w', encoding='utf-8') as converted_file:
            hjson.dump(hyper_config, converted_file)
        return XpotAdapter(converted_file_path, max_iter, **kwargs)

    def get_config_section(self, section_name: str) -> ConfigDict:
        if section_name not in self.config_data:
            raise ValueError(f'No {section_name} configuration found in the config file.')
        return patify(self.config_data[section_name])

    def get_bench_config(self) -> BenchConfig:
        if INF_NAME not in self.config_data:
            raise ValueError('No benchmark configuration found in the config file.')
        return BenchConfig(
            Path(str(self.get_config_section(GEN_NAME)[LMP_BIN_NAME])),
            int(str(self.get_config_section(INF_NAME)[PRE_STEPS_NAME])),
            int(str(self.get_config_section(INF_NAME)[MAX_STEPS_NAME])),
            int(str(self.get_config_section(INF_NAME)[N_CPU_NAME]))
        )

    def get_prop_config(self) -> PropConfig:
        if DA_NAME not in self.config_data:
            raise ValueError('No data analysis configuration found in the config file.')
        return PropConfig(
            Path(str(self.get_config_section(GEN_NAME)[LMP_BIN_NAME])),
            Path(str(self.get_config_section(DA_NAME)[LAMMPS_INPS_NAME])),
            Path(str(self.get_config_section(DA_NAME)[PPS_PYTHON_NAME])),
            Path(str(self.get_config_section(DA_NAME)[REF_DATA_NAME]))
        )
