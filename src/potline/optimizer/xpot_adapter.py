"""
XPOT adapter for the optimization pipeline.
"""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import hjson # type: ignore
from xpot.models import PACE # type: ignore
from xpot.optimiser import NamedOptimiser # type: ignore
from skopt.space import Dimension # type: ignore

from .optimizer import Optimizer

class XpotModel(ABC):
    """
    Interface for the XPOT supported models.

    Args:
        config_path (Path): The path to the configuration file.
    """
    @abstractmethod
    def __init__(self, config_path: Path):
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
    def convert_yace(self, pot_path: Path, out_path: Path) -> Path:
        pass

    @abstractmethod
    def get_optimization_space(self) -> dict[tuple[str, ...], Dimension]:
        pass

    @abstractmethod
    def get_sweep_path(self) -> Path:
        pass

def XpotModelFactory(config_path: Path) -> XpotModel:
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
        config_path (Path): The path to the configuration file.
    """
    def __init__(self, config_path: Path):
        self.model: PACE = PACE(config_path)

    def fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        filename: str = 'xpot-ace.yaml'
    ) -> float:
        return self.model.fit(opt_values, iteration, filename)

    def convert_yace(self, pot_path: Path, out_path: Path) -> Path:
        subprocess.run(['pace_yaml2yace', '-o', out_path, pot_path], check=True)
        return out_path

    def get_optimization_space(self) -> dict[tuple[str, ...], Dimension]:
        return self.model.optimisation_space

    def get_sweep_path(self) -> Path:
        return self.model.sweep_path

class XpotAdapter(Optimizer):
    """
    XPOT adapter for the optimization pipeline.

    Args:
        config_path (Path): The path to the configuration file.
        **kwargs: Additional keyword arguments
    """
    def __init__(self, config_path: Path, **kwargs):
        self.model: XpotModel = XpotModelFactory(config_path)
        self.optimizer = NamedOptimiser(self.model.get_optimization_space(),
                                        self.model.get_sweep_path(), **kwargs)

    def optimize(self, max_iter: int, out_yace_path: Path) -> list[Path]:
        """
        Optimizes the potential using the XPOT optimizer.

        Args:
            max_iter (int): The maximum number of iterations.
            out_yace_path (Path): The path to the output directory.

        Returns:
            list[Path]: The paths to the output directories.
        """
        # Run the optimization
        while self.optimizer.iter <= max_iter:
            self.optimizer.run_optimisation(self.model.fit, path = self.model.get_sweep_path())

        # Convert the best potentials to YACE format
        yace_list: list[Path] = []
        model_dirs = [d for d in self.model.get_sweep_path().iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            # Create the output directory
            out_dir_path = out_yace_path / model_dir
            out_dir_path.mkdir(parents=True, exist_ok=True)
            # Convert the best cycle to YACE format
            yace_list.append(self.model.convert_yace(
                model_dir.resolve() / 'interim_potential_best_cycle.yaml',
                out_dir_path / 'pace.yace'))
        return yace_list

    def get_final_results(self):
        self.optimizer.tabulate_final_results(self.model.get_sweep_path())
