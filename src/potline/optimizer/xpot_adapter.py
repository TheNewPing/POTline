"""
XPOT adapter for the optimization pipeline.
"""

from pathlib import Path
import shutil

from xpot.optimiser import NamedOptimiser # type: ignore

from .optimizer import Optimizer, FITTING_DIR_NAME
from .model import XpotModel, XpotModelFactory

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
        self.model: XpotModel = XpotModelFactory(config_path)
        self.optimizer: NamedOptimiser = NamedOptimiser(self.model.get_optimization_space(),
                                        self.model.get_sweep_path(), kwargs)

    def optimize(self, max_iter: int):
        while self.optimizer.iter <= max_iter:
            self.optimizer.run_optimisation(self.model.fit, path=self.model.get_sweep_path())

    def get_sweep_path(self) -> Path:
        return self.model.get_sweep_path()

    def get_final_results(self) -> None:
        self.optimizer.tabulate_final_results(self.model.get_sweep_path())

        # Move the sweep directory to the fitting directory
        sweep_path: Path = self.model.get_sweep_path()
        fitted_dirs: list[Path] = [item for item in sweep_path.iterdir() if item.is_dir()]
        fitting_dir: Path = sweep_path / FITTING_DIR_NAME
        fitting_dir.mkdir(exist_ok=True)
        for f_dir in fitted_dirs:
            if f_dir.is_dir():
                shutil.move(f_dir, fitting_dir / f_dir.name)
