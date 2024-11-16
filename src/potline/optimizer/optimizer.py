"""
Optimizer interface for the optimization pipeline.
"""

from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Interface for the optimizer.

    Args:
        config_path (str): The path to the configuration
        **kwargs: Additional keyword arguments
    """
    @abstractmethod
    def __init__(self, config_path: str, **kwargs):
        pass

    @abstractmethod
    def optimize(self, max_iter: int, out_yace_path: str) -> list[str]:
        pass

    @abstractmethod
    def get_final_results(self):
        pass
