"""
Potential optimization pipeline API.
"""

from pathlib import Path
from typing import Optional

from .optimizer import Optimizer, XpotAdapter
from .lammps_runner import run_benchmark
from .lammps_analysis import run_properties_simulation
from .utils import get_best_models, convert_yace, create_potential, POTENTIAL_NAME
from .config_reader import ConfigReader, GEN_NAME, MODEL_NAME, BEST_N_NAME

class PotLine():
    """
    Main class for running the optimization pipeline.

    Args:
    - config_path: path to the configuration file.
    - with_hyper_search: flag to run the hyperparameter search.
    - with_conversion: flag to convert the results to LAMMPS format.
    - with_inference: flag to run the inference benchmark.
    - with_data_analysis: flag to run the data analysis on mechanical properties.
    - hpc: flag to run the simulations on HPC.
    - fitted_path: path to the directory with the fitted models.
    """
    def __init__(self,
                 config_path: Path,
                 with_hyper_search: bool,
                 with_conversion: bool,
                 with_inference: bool,
                 with_data_analysis: bool,
                 hpc: bool,
                 fitted_path: Optional[Path] = None):
        self.config_reader: ConfigReader = ConfigReader(config_path)
        self.with_hyper_search: bool = with_hyper_search
        self.with_conversion: bool = with_conversion
        self.with_inference: bool = with_inference
        self.with_data_analysis: bool = with_data_analysis
        self.hpc: bool = hpc
        self.model_name: str = str(self.config_reader.get_config_section(GEN_NAME)[MODEL_NAME])
        self.best_n_models: int = int(str(self.config_reader.get_config_section(GEN_NAME)[BEST_N_NAME]))
        if self.with_hyper_search:
            self.optimizer: Optimizer = XpotAdapter(*self.config_reader.get_optimizer_config())
        self.fitted_path: Path = fitted_path if fitted_path else self.optimizer.get_sweep_path()

    def run(self) -> None:
        """
        Run the optimization pipeline.
        1. Optimize the potential, convert the results to yace format, print the final results.
        2. Run the inference benchmark.
        3. Run the data analysis on mechanical properties
        """
        if self.with_hyper_search:
            self.optimizer.optimize()
            self.optimizer.get_final_results()

        if self.with_conversion:
            yace_list = convert_yace(self.model_name, self.fitted_path)
        else:
            yace_list = get_yaces(self.fitted_path)

        yace_list = get_best_models(self.fitted_path, yace_list, self.best_n_models)

        for yace_path in yace_list:
            create_potential(self.model_name, yace_path, yace_path.parent)

        if self.with_inference:
            for yace_path in yace_list:
                run_benchmark(yace_path.parent, self.config_reader.get_bench_config(), hpc=self.hpc)

        if self.with_data_analysis:
            for yace_path in yace_list:
                run_properties_simulation(yace_path.parent,
                                          self.config_reader.get_prop_config(), hpc=self.hpc)

def get_yaces(out_yace_path: Path) -> list[Path]:
    """
    Get the list of yace files in the output directory.

    Args:
    - out_yace_path: path to the output directory.

    Returns:
    - list of yace files.
    """
    return list(out_yace_path.glob('*.yace'))

def get_potentials(fitted_path: Path) -> list[Path]:
    """
    Get the list of potential files in the fitted directory.

    Args:
    - fitted_path: path to the fitted directory.

    Returns:
    - list of potential files.
    """
    return list(fitted_path.glob(POTENTIAL_NAME))
