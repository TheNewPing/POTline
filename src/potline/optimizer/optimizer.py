"""
Optimizer interface for the optimization pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path

class Optimizer(ABC):
    """
    Interface for the optimizer.
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
