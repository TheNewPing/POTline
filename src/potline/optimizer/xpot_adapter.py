"""
XPOT adapter for the optimization pipeline.
"""

from pathlib import Path

from .optimizer import Optimizer
from .model import HPCMLP, create_xpot_model
from .hpc_optimizer import HPCOptimizer
from ..config_reader import HyperConfig

class XpotAdapter(Optimizer):
    """
    XPOT adapter for the optimization pipeline.

    Args:
        config_path (Path): The path to the configuration file.
        **kwargs: Additional keyword arguments
    """
    def __init__(self, config_path: Path, config: HyperConfig):
        self.model: HPCMLP = create_xpot_model(config_path)
        self.max_iter: int = config.max_iter
        self.optimizer: HPCOptimizer = HPCOptimizer(self.model.optimisation_space,
                                        self.model.sweep_path, config.n_points, config.strategy,
                                        {"n_initial_points": config.n_initial_points})

    def optimize(self):
        while self.optimizer.iter <= self.max_iter:
            self.optimizer.run_hpc_optimization(self.model.dispatch_fit,
                                                self.model.collect_loss, path=self.model.sweep_path)

    def get_sweep_path(self) -> Path:
        return Path(self.model.sweep_path)

    def get_final_results(self) -> None:
        self.optimizer.tabulate_final_results(self.model.sweep_path)
        #self.optimizer.plot_results(self.model.sweep_path)
