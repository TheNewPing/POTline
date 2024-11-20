"""
XpotModel interface and factory function.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import hjson # type: ignore
from xpot.models import PACE # type: ignore
from skopt.space import Dimension # type: ignore

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

    def get_optimization_space(self) -> dict[tuple[str, ...], Dimension]:
        return self.model.optimisation_space

    def get_sweep_path(self) -> Path:
        return Path(self.model.sweep_path)
