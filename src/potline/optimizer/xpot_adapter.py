"""
XPOT adapter for the optimization pipeline.
"""

import os
import subprocess
from abc import ABC, abstractmethod

import hjson # type: ignore
from xpot.models import PACE # type: ignore
from xpot.optimiser import NamedOptimiser # type: ignore
from skopt.space import Dimension # type: ignore

from .optimizer import Optimizer

class XpotModel(ABC):
    """
    Interface for the XPOT supported models.

    Args:
        config_path (str): The path to the configuration file.
    """
    @abstractmethod
    def __init__(self, config_path: str):
        pass

    @abstractmethod
    def fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        filename: str
    ) -> float:
        pass

    @abstractmethod
    def convert_yace(self, pot_path: str, out_path: str) -> str:
        pass

    @abstractmethod
    def get_optimization_space(self) -> dict[tuple[str, ...], Dimension]:
        pass

    @abstractmethod
    def get_sweep_path(self) -> str:
        pass

def XpotModelFactory(config_path: str) -> XpotModel:
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data: dict = hjson.load(file)
        if config_data['xpot']['fitting_executable'] == 'pacemaker':
            return XpotPACE(config_path)
        else:
            raise ValueError('Model not supported.')

class XpotPACE(XpotModel):
    """
    XPOT model for the PACE optimizer.

    Args:
        config_path (str): The path to the configuration file.
    """
    def __init__(self, config_path: str):
        self.model: PACE = PACE(config_path)

    def fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        filename: str = 'xpot-ace.yaml'
    ) -> float:
        return self.model.fit(opt_values, iteration, filename)

    def convert_yace(self, pot_path: str, out_path: str) -> str:
        subprocess.run(['pace_yaml2yace', '-o', out_path, pot_path], check=True)
        return os.path.join(os.getcwd(), 'pace.yace')

    def get_optimization_space(self) -> dict[tuple[str, ...], Dimension]:
        return self.model.optimisation_space

    def get_sweep_path(self) -> str:
        return self.model.sweep_path

class XpotAdapter(Optimizer):
    """
    XPOT adapter for the optimization pipeline.

    Args:
        config_path (str): The path to the configuration file.
        **kwargs: Additional keyword arguments
    """
    def __init__(self, config_path: str, **kwargs):
        self.model: XpotModel = XpotModelFactory(config_path)
        self.optimizer = NamedOptimiser(self.model.get_optimization_space(),
                                        self.model.get_sweep_path(), **kwargs)

    def optimize(self, max_iter: int, out_yace_path: str) -> list[str]:
        """
        Optimizes the potential using the XPOT optimizer.

        Args:
            max_iter (int): The maximum number of iterations.
            out_yace_path (str): The path to the output directory.

        Returns:
            list[str]: The paths to the output directories.
        """
        # Run the optimization
        while self.optimizer.iter <= max_iter:
            self.optimizer.run_optimisation(self.model.fit, path = self.model.get_sweep_path())

        # Convert the best potentials to YACE format
        yace_list: list[str] = []
        for folder in os.listdir(self.model.get_sweep_path()):
            folder_path = os.path.join(self.model.get_sweep_path(), folder)
            if os.path.isdir(folder_path):
                out_folder_path = os.path.join(out_yace_path, folder)
                os.makedirs(out_folder_path, exist_ok=True)
                pot_path = os.path.join(folder_path, 'interim_potential_best_cycle.yaml')
                out_pot_path = os.path.join(out_folder_path, 'pace.yace')
                self.model.convert_yace(pot_path, out_pot_path)
                yace_list.append(out_pot_path)
        return yace_list

    def get_final_results(self):
        self.optimizer.tabulate_final_results(self.model.get_sweep_path())
