"""
DeepTrainer class for training after hyperparameter search.
"""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

class DeepTrainer(ABC):
    """
    Abstract class for training after hyperparameter search.

    Args:
    - max_epochs: maximum number of epochs for training.
    - config_filepath: path to the configuration file.
    - potential_filepath: path to the potential file.
    """
    def __init__(self, max_epochs: int, config_filepath: Path, potential_filepath: Path):
        self.max_epochs = max_epochs
        self.config_filepath = config_filepath
        self.potential_filepath = potential_filepath

    @abstractmethod
    def setup_config(self) -> Path:
        """
        Update the configuration file with the maximum number of epochs.
        """

    @abstractmethod
    def train(self) -> None:
        """
        Train the model.
        """

def create_deep_trainer(modelname: str,
                        max_epochs: int,
                        config_filepath: Path,
                        potential_filepath: Path) -> DeepTrainer:
    """
    Create a DeepTrainer object based on the desired model.

    Args:
    - modelname: name of the model (pacemaker, ...).
    - max_epochs: maximum number of epochs for training.
    - config_filepath: path to the configuration file.
    - potential_filepath: path to the potential file.

    Returns:
    - DeepTrainer object.
    """
    if modelname == 'pacemaker':
        return PACEDeepTrainer(max_epochs, config_filepath, potential_filepath)
    raise ValueError('Model not supported.')

class PACEDeepTrainer(DeepTrainer):
    """
    DeepTrainer class for the PACEMAKER model.
    """
    def setup_config(self):
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

    def train(self) -> None:
        config = self.setup_config()
        subprocess.run(['pacemaker', str(config), '-p', str(self.potential_filepath)], check=True)
