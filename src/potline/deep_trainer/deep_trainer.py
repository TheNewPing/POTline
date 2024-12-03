"""
DeepTrainer class for training after hyperparameter search.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from simple_slurm import Slurm

from ..optimizer.model import CONFIG_NAME
from ..utils import LAST_POTENTIAL_NAME
from ..config_reader import DeepTrainConfig

class DeepTrainer(ABC):
    """
    Abstract class for training after hyperparameter search.

    Args:
        - max_epochs: maximum number of epochs for training.
        - out_path: path to the output directory.
    """
    def __init__(self, max_epochs: int, out_path: Path):
        self.max_epochs = max_epochs
        self.config_filepath = out_path / CONFIG_NAME
        self.potential_filepath = out_path / LAST_POTENTIAL_NAME

    @abstractmethod
    def setup_config(self) -> Path:
        pass

    @abstractmethod
    def dispatch_train(self) -> int:
        pass

def create_deep_trainer(config: DeepTrainConfig, out_path: Path) -> DeepTrainer:
    """
    Factory function for creating a DeepTrainer object.

    Args:
        - config: configuration for the deep training.
        - out_path: path to the output directory.

    Returns:
        DeepTrainer: the DeepTrainer object.
    """
    if config.model_name == 'pacemaker':
        return PACEHPCDeepTrainer(config.max_epochs, out_path)
    raise ValueError('Model not supported.')

class PACEHPCDeepTrainer(DeepTrainer):
    """
    DeepTrainer class for the PACEMAKER model.
    Requires Slurm.
    """
    def setup_config(self):
        os.chdir(self.config_filepath.parent)
        # Read the YAML file
        with self.config_filepath.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # Update the maxiter value
        if 'fit' in config and 'maxiter' in config['fit']:
            config['fit']['maxiter'] = self.max_epochs
        else:
            raise KeyError("The key 'fit' or 'maxiter' does not exist in the YAML file.")

        # Write the updated config to a new file
        output_filepath = self.config_filepath.with_stem(self.config_filepath.stem + "_deep_train")
        with output_filepath.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)

        return output_filepath

    def dispatch_train(self) -> int:
        config = self.setup_config()
        command: list[str] = ['pacemaker', str(config), '-p', str(self.potential_filepath)]
        out_path: Path = self.potential_filepath.parent
        fit_job = Slurm(
            job_name="deep_pace",
            output=f"{out_path}/deep_%j.out",
            error=f"{out_path}/deep_%j.err",
            time="36:00:00",
            mem="50G",
            partition="gpu",
            nodes=1,
            ntasks=1,
            cpus_per_task=16,
            gpus=1,
        )
        fit_job.add_cmd("module load 2024")
        fit_job.add_cmd("module load Miniconda3/24.7.1-0")
        fit_job.add_cmd("module load 2022")
        fit_job.add_cmd("module load cuDNN/8.4.1.50-CUDA-11.7.0")
        fit_job.add_cmd("export LD_LIBRARY_PATH=/home/erodaro/.conda/envs/pl/lib/:$LD_LIBRARY_PATH")
        fit_job.add_cmd("source $(conda info --base)/etc/profile.d/conda.sh")
        fit_job.add_cmd("conda activate pl")
        return fit_job.sbatch(' '.join(command))
