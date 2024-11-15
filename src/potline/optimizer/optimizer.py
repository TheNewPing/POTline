from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __init__(self, config_path: str, **kwargs):
        pass

    @abstractmethod
    def optimize(self, max_iter: int):
        pass

    @abstractmethod
    def get_final_results(self):
        pass

    @abstractmethod
    def convert_yace(self, pot_path: str, out_path: str) -> str:
        pass