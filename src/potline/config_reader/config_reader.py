"""
Configuration file reader for the optimization pipeline.
"""
from typing import Any
from pathlib import Path
from enum import Enum

import hjson # type: ignore

class MainSectionKW(Enum):
    """
    Main sections of the configuration file.
    """
    GENERAL = 'general'
    DEEP_TRAINING = 'deep_training'
    INFERENCE = 'inference'
    PROP_SIM = 'data_analysis'
    HYPER_SEARCH = 'hyper_search'

class SlurmJobKW(Enum):
    """
    Keywords for the slurm job configuration.
    """
    SLURM_WATCHER = 'slurm_watcher'
    SLURM_OPTS = 'slurm_opts'
    MODULES = 'modules'
    PY_SCRIPTS = 'py_scripts'

class GeneralKW(Enum):
    """
    General keywords for the configuration file.
    """
    LMP_BIN = 'lammps_bin_path'
    MODEL = 'model_name'
    BEST_N = 'best_n_models'
    HPC = 'hpc'
    CLUSTER = 'cluster'
    SWEEP_PATH = 'sweep_path'

class DeepTrainKW(Enum):
    """
    Deep training keywords for the configuration file.
    """
    MAX_EPOCHS = 'max_epochs'

class InferenceKW(Enum):
    """
    Inference keywords for the configuration file.
    """
    PRE_STEPS = 'prerun_steps'
    MAX_STEPS = 'max_steps'

class PropSimKW(Enum):
    """
    Keywords for the property simulation configuration.
    """
    LAMMPS_INPS = 'lammps_inps_path'
    PPS_PYTHON = 'pps_python_path'
    REF_DATA = 'ref_data_path'
    EMAIL = 'email'

class HyperSearchKW(Enum):
    """
    Keywords for the hyperparameter search configuration.
    """
    MAX_ITER = 'max_iter'
    N_INIT_PTS = 'n_initial_points'
    N_PTS = 'n_points'
    STRAT = 'strategy'
    ENERGY_WEIGHT = 'energy_weight'
    OPTIMIZER_PARAMS = 'optimizer_params'

class JobConfig():
    """
    Configuration class for the job configuration.
    """
    def __init__(self, slurm_watcher: dict,
                 slurm_opts: dict,
                 modules: list[str],
                 py_scripts: list[str],
                 cluster: str):
        self.slurm_watcher: dict = slurm_watcher
        self.slurm_opts: dict = slurm_opts
        self.modules: list[str] = modules
        self.py_scripts: list[str] = py_scripts
        self.cluster: str = cluster

class BenchConfig():
    """
    Configuration class for the benchmarking step.
    """
    def __init__(self, lammps_bin_path: Path,
                 prerun_steps: int,
                 max_steps: int,
                 sweep_path: Path,
                 job_config: JobConfig):
        self.lammps_bin_path: Path = lammps_bin_path
        self.prerun_steps: int = prerun_steps
        self.max_steps: int = max_steps
        self.sweep_path: Path = sweep_path
        self.job_config: JobConfig = job_config

class PropConfig():
    """
    Configuration class for the data analysis step.
    """
    def __init__(self, lammps_bin_path: Path,
                 lammps_inps_path: Path,
                 pps_python_path: Path,
                 ref_data_path: Path,
                 sweep_path: Path,
                 job_config: JobConfig):
        self.lammps_bin_path: Path = lammps_bin_path
        self.lammps_inps_path: Path = lammps_inps_path
        self.pps_python_path: Path = pps_python_path
        self.ref_data_path: Path = ref_data_path
        self.sweep_path: Path = sweep_path
        self.job_config: JobConfig = job_config

class HyperConfig():
    """
    Configuration class for the hyperparameter search.
    """
    def __init__(self, model_name: str,
                 sweep_path: Path,
                 max_iter: int,
                 n_initial_points: int,
                 n_points: int,
                 strategy: str,
                 energy_weight: float,
                 optimizer_params: dict,
                 job_config: JobConfig,):
        self.model_name: str = model_name
        self.sweep_path: Path = sweep_path
        self.max_iter: int = max_iter
        self.n_initial_points: int = n_initial_points
        self.n_points: int = n_points
        self.strategy: str = strategy
        self.energy_weight: float = energy_weight
        self.optimizer_params: dict = optimizer_params
        self.job_config: JobConfig = job_config

class DeepTrainConfig():
    """
    Configuration class for the deep training step.
    """
    def __init__(self, max_epochs: int,
                 sweep_path: Path,
                 job_config: JobConfig):
        self.max_epochs: int = max_epochs
        self.sweep_path: Path = sweep_path
        self.job_config: JobConfig = job_config

class GeneralConfig():
    """
    Configuration class for the general configuration.
    """
    def __init__(self, lammps_bin_path: Path,
                 model_name: str,
                 best_n_models: int,
                 hpc: bool,
                 cluster: str,
                 sweep_path: Path,
                 job_config: JobConfig):
        self.lammps_bin_path: Path = lammps_bin_path
        self.model_name: str = model_name
        self.best_n_models: int = best_n_models
        self.hpc: bool = hpc
        self.cluster: str = cluster
        self.sweep_path: Path = sweep_path
        self.job_config: JobConfig = job_config

def patify(config_dict: dict[str, Any]) -> dict:
    """
    Convert all string path values in the dictionary to Path objects.
    """
    for key, value in config_dict.items():
        if key.endswith('_path') and isinstance(value, str):
            config_dict[key] = Path(value)
    return config_dict

class ConfigReader():
    """
    Class for reading and converting configuration files.
    A config file is written in hjson format and should have the following main sections:
    - general: general configuration for the pipeline.
    - hyper_search: configuration for the optimizer, uses the XPOT format.
    - deep_training: configuration for the deep training after the hyperparameter search.
    - inference: configuration for the inference benchmark with LAMMPS.
    - data_analysis: configuration for the data analysis on mechanical properties with LAMMPS.
    """
    def __init__(self, file_path: Path):
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f'Configuration file {file_path} not found.')
        self.file_path: Path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            self.config_data: dict = hjson.load(file)

    def get_config_section(self, section_name: str) -> dict:
        """
        Get a specific section from the configuration file.
        """
        if section_name not in self.config_data:
            raise ValueError(f'No {section_name} configuration found in the config file.')
        return patify(self.config_data[section_name])

    def get_slurm_config(self, section_name: str) -> JobConfig:
        """
        Get the SLURM configuration for a specific job type.
        """
        gen_config: dict = self.get_config_section(MainSectionKW.GENERAL.value)
        section_config: dict = self.get_config_section(section_name)
        return JobConfig(
            section_config[SlurmJobKW.SLURM_WATCHER.value],
            section_config[SlurmJobKW.SLURM_OPTS.value],
            section_config[SlurmJobKW.MODULES.value],
            section_config[SlurmJobKW.PY_SCRIPTS.value],
            gen_config[GeneralKW.CLUSTER.value]
        )

    def get_optimizer_config(self) -> HyperConfig:
        """
        Get the optimizer configuration from the configuration file.
        """
        if MainSectionKW.HYPER_SEARCH.value not in self.config_data:
            raise ValueError('No hyperparameter search configuration found in the config file.')
        return HyperConfig(
            str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.MODEL.value]),
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.SWEEP_PATH.value])),
            int(str(self.get_config_section(MainSectionKW.HYPER_SEARCH.value)[HyperSearchKW.MAX_ITER.value])),
            int(str(self.get_config_section(
                MainSectionKW.HYPER_SEARCH.value)[HyperSearchKW.N_INIT_PTS.value])),
            int(str(self.get_config_section(MainSectionKW.HYPER_SEARCH.value)[HyperSearchKW.N_PTS.value])),
            str(self.get_config_section(MainSectionKW.HYPER_SEARCH.value)[HyperSearchKW.STRAT.value]),
            float(str(self.get_config_section(
                MainSectionKW.HYPER_SEARCH.value)[HyperSearchKW.ENERGY_WEIGHT.value])),
            self.get_config_section(MainSectionKW.HYPER_SEARCH.value)[HyperSearchKW.OPTIMIZER_PARAMS.value],
            self.get_slurm_config(MainSectionKW.HYPER_SEARCH.value)
        )

    def get_bench_config(self) -> BenchConfig:
        if MainSectionKW.INFERENCE.value not in self.config_data:
            raise ValueError('No benchmark configuration found in the config file.')
        return BenchConfig(
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.LMP_BIN.value])),
            int(str(self.get_config_section(MainSectionKW.INFERENCE.value)[InferenceKW.PRE_STEPS.value])),
            int(str(self.get_config_section(MainSectionKW.INFERENCE.value)[InferenceKW.MAX_STEPS.value])),
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.SWEEP_PATH.value])),
            self.get_slurm_config(MainSectionKW.INFERENCE.value)
        )

    def get_prop_config(self) -> PropConfig:
        if MainSectionKW.PROP_SIM.value not in self.config_data:
            raise ValueError('No property simulation configuration found in the config file.')
        return PropConfig(
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.LMP_BIN.value])),
            Path(str(self.get_config_section(MainSectionKW.PROP_SIM.value)[PropSimKW.LAMMPS_INPS.value])),
            Path(str(self.get_config_section(MainSectionKW.PROP_SIM.value)[PropSimKW.PPS_PYTHON.value])),
            Path(str(self.get_config_section(MainSectionKW.PROP_SIM.value)[PropSimKW.REF_DATA.value])),
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.SWEEP_PATH.value])),
            self.get_slurm_config(MainSectionKW.PROP_SIM.value)
        )

    def get_deep_train_config(self) -> DeepTrainConfig:
        if MainSectionKW.DEEP_TRAINING.value not in self.config_data:
            raise ValueError('No deep training configuration found in the config file.')
        return DeepTrainConfig(
            int(str(self.get_config_section(MainSectionKW.DEEP_TRAINING.value)[DeepTrainKW.MAX_EPOCHS.value])),
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.SWEEP_PATH.value])),
            self.get_slurm_config(MainSectionKW.DEEP_TRAINING.value)
        )

    def get_general_config(self) -> GeneralConfig:
        if MainSectionKW.GENERAL.value not in self.config_data:
            raise ValueError('No general configuration found in the config file.')
        return GeneralConfig(
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.LMP_BIN.value])),
            str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.MODEL.value]),
            int(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.BEST_N.value])),
            bool(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.HPC.value])),
            str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.CLUSTER.value]),
            Path(str(self.get_config_section(MainSectionKW.GENERAL.value)[GeneralKW.SWEEP_PATH.value])),
            self.get_slurm_config(MainSectionKW.GENERAL.value)
        )
