"""
Potential optimization pipeline API.
"""

from pathlib import Path

from .hyper_searcher import PotOptimizer
from .inference_bencher import run_benchmark
from .properties_simulator import run_properties_simulation
from .config_reader import ConfigReader
from .deep_trainer import DeepTrainer
from .model import ModelTracker, PotModel, POTENTIAL_NAME
from .dispatcher import DispatcherFactory, JobType

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
    """
    def __init__(self,
                 config_path: Path,
                 with_hyper_search: bool,
                 with_deep_train: bool,
                 with_conversion: bool,
                 with_inference: bool,
                 with_data_analysis: bool):
        self.config_reader: ConfigReader = ConfigReader(config_path)
        self.with_hyper_search: bool = with_hyper_search
        self.with_deep_train: bool = with_deep_train
        self.with_conversion: bool = with_conversion
        self.with_inference: bool = with_inference
        self.with_data_analysis: bool = with_data_analysis
        self.config = self.config_reader.get_general_config()
        self.optimizer = PotOptimizer(self.config_reader.get_optimizer_config(),
                                      DispatcherFactory(JobType.FIT,
                                                        self.config.cluster))

    def run(self):
        optimized_models: list[ModelTracker] = self.hyper_search()
        best_models: list[ModelTracker] = self.filter_best_loss(optimized_models)
        deep_models: list[ModelTracker] = self.deep_train(best_models)
        models_to_test: list[PotModel] = [model.model for model in deep_models]
        models_out_path: list[Path] = self.prepare_lammps(models_to_test)
        self.inference_bench(models_out_path)
        self.properties_simulation(models_out_path)

    def hyper_search(self) -> list[ModelTracker]:
        if self.with_hyper_search:
            return self.optimizer.run()
        raise ValueError("Hyperparameter search not enabled.")

    def filter_best_loss(self, model_list: list[ModelTracker]) -> list[ModelTracker]:
        sorted_models = sorted(model_list, key=lambda model: model.get_total_test_loss(
            self.config_reader.get_optimizer_config().energy_weight))
        return sorted_models[:self.config.best_n_models]

    def deep_train(self, model_list: list[ModelTracker]) -> list[ModelTracker]:
        """
        Additional training for the best models.
        """
        if self.with_deep_train:
            deep_trainers: list[DeepTrainer] = []
            deep_models: list[ModelTracker] = []
            for model in model_list:
                deep_trainer = DeepTrainer(self.config_reader.get_deep_train_config(), model,
                                           DispatcherFactory(JobType.DEEP, self.config.cluster))
                deep_trainer.run()
                deep_trainers.append(deep_trainer)
            for trainer in deep_trainers:
                deep_models.append(trainer.collect())
        return deep_models

    def prepare_lammps(self, model_list: list[PotModel]) -> list[Path]:
        out: list[Path] = []
        if self.with_conversion:
            for model in model_list:
                model.lampify()
                model.create_potential()
                out.append(model.get_out_path())
        return out

    def inference_bench(self, pot_list: list[Path]):
        if self.with_inference:
            for pot_path in pot_list:
                run_benchmark(pot_path.parent, self.config_reader.get_bench_config(),
                              DispatcherFactory(JobType.INF, self.config.cluster))

    def properties_simulation(self, pot_list: list[Path]):
        if self.with_data_analysis:
            for pot_path in pot_list:
                run_properties_simulation(pot_path.parent,
                                          self.config_reader.get_prop_config(),
                                          DispatcherFactory(JobType.SIM, self.config.cluster))

    def get_yaces(self) -> list[Path]:
        return list(self.config.sweep_path.glob('*.yace'))

    def get_potentials(self) -> list[Path]:
        return list(self.config.sweep_path.glob(POTENTIAL_NAME))
