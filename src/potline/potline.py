"""
Potential optimization pipeline API.
"""

from pathlib import Path
from typing import Optional

from .config_reader import ConfigReader
from .optimizer import Optimizer
from .optimizer.model import convert_yace
from .lammps_runner import run_benchmark
from .lammps_analysis import run_properties_simulation

def get_yaces(out_yace_path: Path) -> list[Path]:
    """
    Get the list of yace files in the output directory.

    Args:
    - out_yace_path: path to the output directory.

    Returns:
    - list of yace files.
    """
    return list(out_yace_path.glob('*.yace'))

class PotLine():
    """
    Main class for running the optimization pipeline.

    Args:
    - config_path: path to the configuration file.
    - max_iter: maximum number of iterations for the optimization.
    - with_inference: flag to run the inference benchmark.
    - with_data_analysis: flag to run the data analysis on mechanical properties.
    """
    def __init__(self, config_path: Path,
                 max_iter: int,
                 with_fitting: bool,
                 with_conversion: bool,
                 with_inference: bool,
                 with_data_analysis: bool,
                 fitted_path: Optional[Path] = None):

        self.config_reader = ConfigReader(config_path)
        self.max_iter: int = max_iter
        self.with_fitting: bool = with_fitting
        self.with_conversion: bool = with_conversion
        self.with_inference: bool = with_inference
        self.with_data_analysis: bool = with_data_analysis
        self.lammps_bin_path: Path = self.config_reader.get_lammps_bin_path()
        self.out_yace_path: Path = self.config_reader.get_out_yace_path()
        self.model_name: str = self.config_reader.get_model_name()
        if self.with_fitting:
            self.optimizer: Optimizer = self.config_reader.create_optimizer()
        self.fitted_path: Path = fitted_path if fitted_path else self.optimizer.get_sweep_path()

    def run(self) -> None:
        """
        Run the optimization pipeline.
        1. Optimize the potential, convert the results to yace format, print the final results.
        2. Run the inference benchmark.
        3. Run the data analysis on mechanical properties
        """
        if self.with_fitting:
            self.optimizer.optimize(self.max_iter)
            self.optimizer.get_final_results()

        if self.with_conversion:
            yace_list = convert_yace(self.model_name, self.fitted_path, self.out_yace_path)
        else:
            yace_list = get_yaces(self.out_yace_path)

        if self.with_inference:
            inf_config: dict = self.config_reader.get_inf_benchmark_config()
            for yace_path in yace_list:
                run_benchmark(yace_path.parent, yace_path, self.lammps_bin_path, **inf_config)

        if self.with_data_analysis:
            data_config: dict = self.config_reader.get_data_analysis_config()
            for yace_path in yace_list:
                run_properties_simulation(yace_path.parent, yace_path, self.lammps_bin_path, **data_config)
