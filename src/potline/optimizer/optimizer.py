"""
Optimizer interface for the optimization pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path

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
    def optimize(self, max_iter: int, out_yace_path: Path) -> list[Path]:
        pass

    @abstractmethod
    def get_final_results(self):
        pass
