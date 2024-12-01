"""
Optimizer interface for the optimization pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path

BEST_POTENTIAL_NAME: str = 'interim_potential_best_cycle.yaml'

class Optimizer(ABC):
    """
    Interface for the optimizer.

    Args:
        config_path (Path): The path to the configuration
        **kwargs: Additional keyword arguments
    """
    @abstractmethod
    def __init__(self, config_path: Path, **kwargs):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def get_sweep_path(self) -> Path:
        pass

    @abstractmethod
    def get_final_results(self):
        pass
