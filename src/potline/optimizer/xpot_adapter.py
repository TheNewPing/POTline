"""
XPOT adapter for the optimization pipeline.
"""

from pathlib import Path

from xpot.optimiser import NamedOptimiser # type: ignore

from .optimizer import Optimizer
from .model import XpotModel, create_xpot_model

class XpotAdapter(Optimizer):
    """
    XPOT adapter for the optimization pipeline.

    Args:
        config_path (Path): The path to the configuration file.
        **kwargs: Additional keyword arguments
    """
    def __init__(self, config_path: Path, **kwargs):
        kwargs = {
        "n_initial_points": 5,
        }
        self.model: XpotModel = create_xpot_model(config_path)
        self.optimizer: NamedOptimiser = NamedOptimiser(self.model.get_optimization_space(),
                                        self.model.get_sweep_path(), kwargs)

    def optimize(self, max_iter: int):
        while self.optimizer.iter <= max_iter:
            self.optimizer.run_optimisation(self.model.fit, path=self.model.get_sweep_path())

    def get_sweep_path(self) -> Path:
        return self.model.get_sweep_path()

    def get_final_results(self) -> None:
        self.optimizer.tabulate_final_results(self.model.get_sweep_path())
