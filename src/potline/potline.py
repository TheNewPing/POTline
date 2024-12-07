"""
Potential optimization pipeline API.
"""

import time
from pathlib import Path
from typing import Optional

from simple_slurm import Slurm
import pandas as pd

from .hyper_searcher import Optimizer, XpotAdapter
from .inference_bencher import run_benchmark
from .properties_simulator import run_properties_simulation
from .utils import get_best_models, convert_yace, create_potential, POTENTIAL_NAME
from .config_reader import ConfigReader, GEN_NAME, MODEL_NAME, BEST_N_NAME
from .deep_trainer import create_deep_trainer, DeepTrainer

FINAL_REPORT_NAME: str = 'parameters.csv'
LOSS_COL_KW: str = 'loss'
ITER_COL_KW: str = 'iteration'
SUBITER_COL_KW: str = 'subiteration'

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
                 with_deep_train: bool,
                 with_conversion: bool,
                 with_inference: bool,
                 with_data_analysis: bool,
                 hpc: bool,
                 fitted_path: Optional[Path] = None):
        self.config_reader: ConfigReader = ConfigReader(config_path)
        self.with_hyper_search: bool = with_hyper_search
        self.with_deep_train: bool = with_deep_train
        self.with_conversion: bool = with_conversion
        self.with_inference: bool = with_inference
        self.with_data_analysis: bool = with_data_analysis
        self.hpc: bool = hpc
        self.best_n_models: int = int(str(self.config_reader.get_config_section(GEN_NAME)[BEST_N_NAME]))
        self.model: PotModel
        self.hyper_searcher: PotSearcher
        self.experiment_path: Path 

    def run(self):
        pot_list: list[Path] = self.hyper_search()
        best_pots: list[Path] = self.filter_best_loss(pot_list)
        deep_pot_list: list[Path] = self.deep_train(best_pots)
        self.prepare_lammps(deep_pot_list)
        self.inference_bench(deep_pot_list)
        self.properties_simulation(deep_pot_list)

    def hyper_search(self) -> list[Path]:
        if not self.with_hyper_search:
            return self.get_potentials()
        return self.optimizer.optimize()

    def filter_best_loss(self, filepath_list: list[Path]) -> list[Path]:
        df: pd.DataFrame = pd.read_csv(self.experiment_path / FINAL_REPORT_NAME)
        best_iterations_rows: pd.DataFrame = df.nsmallest(self.best_n_models, LOSS_COL_KW)[[ITER_COL_KW, SUBITER_COL_KW]]
        best_iterations: list[tuple[int, int]] = [(int(row[ITER_COL_KW]), int(row[SUBITER_COL_KW]))
                                                for _, row in best_iterations_rows.iterrows()]
        return [fp for fp in filepath_list if
                (int(fp.parent.parent.name), int(fp.parent.name)) in best_iterations]

    def deep_train(self, pot_list: list[Path]) -> list[Path]:
        if self.with_deep_train:
            # Dispatch the deep training jobs
            deep_id_list: list[int] = []
            for pot_path in pot_list:
                deep_trainer: DeepTrainer = create_deep_trainer(
                    self.config_reader.get_deep_train_config(), pot_path.parent)
                deep_id_list.append(deep_trainer.dispatch_train())
            # Wait for the deep training jobs to finish
            wait_job = Slurm()
            while True:
                wait_job.squeue.update_squeue()
                if not any(wait_id in wait_job.squeue.jobs for wait_id in deep_id_list):
                    break
                time.sleep(10)
        return pot_list

    def prepare_lammps(self, pot_list: list[Path]) -> None:
        if self.with_conversion:
            yace_list = convert_yace(self.model_name, pot_list)
        else:
            yace_list = self.get_yaces()

        for yace_path in yace_list:
            create_potential(self.model_name, yace_path, yace_path.parent)

    def inference_bench(self, pot_list: list[Path]) -> None:
        if self.with_inference:
            for pot_path in pot_list:
                run_benchmark(pot_path.parent, self.config_reader.get_bench_config(), hpc=self.hpc)

    def properties_simulation(self, pot_list: list[Path]) -> None:
        if self.with_data_analysis:
            for pot_path in pot_list:
                run_properties_simulation(pot_path.parent,
                                          self.config_reader.get_prop_config(), hpc=self.hpc)

    def get_yaces(self) -> list[Path]:
        return list(self.experiment_path.glob('*.yace'))

    def get_potentials(self) -> list[Path]:
        return list(self.experiment_path.glob(POTENTIAL_NAME))
