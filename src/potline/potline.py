"""
Potential optimization pipeline API.
"""

from pathlib import Path

from .hyper_searcher import PotOptimizer, OPTIM_DIR_NAME
from .inference_bencher import InferenceBencher
from .properties_simulator import PropertiesSimulator
from .config_reader import ConfigReader
from .deep_trainer import DeepTrainer, DEEP_TRAIN_DIR_NAME
from .model import PotModel
from .dispatcher import DispatcherManager, JobType
from .loss_logger import ModelTracker

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

    def run(self):
        optimized_models: list[ModelTracker] = self.hyper_search()
        best_models: list[ModelTracker] = self.filter_best_loss(optimized_models)
        deep_models: list[ModelTracker] = self.deep_train(best_models)
        models_to_test: list[PotModel] = [model.model for model in deep_models]
        lamped_models: list[Path] = self.prepare_lammps(models_to_test)
        self.inference_bench(lamped_models)
        self.properties_simulation(lamped_models)

    def hyper_search(self) -> list[ModelTracker]:
        """
        Run the hyperparameter search.
        """
        if self.with_hyper_search:
            return PotOptimizer(self.config_reader.get_optimizer_config(),
                                DispatcherManager(JobType.FIT.value, self.config.model_name,
                                                  self.config.cluster)
                                ).run()

        models: list[ModelTracker] = []
        for model in self.get_model_out_paths():
            if model.is_dir():
                model_tracker = ModelTracker.from_path(
                    self.config.model_name, model, self.config.sweep_path)
                models.append(model_tracker)
        return models

    def filter_best_loss(self, model_list: list[ModelTracker]) -> list[ModelTracker]:
        sorted_models = sorted(model_list, key=lambda model: model.get_total_valid_loss(
            self.config_reader.get_optimizer_config().energy_weight))
        return sorted_models[:self.config.best_n_models]

    def deep_train(self, model_list: list[ModelTracker]) -> list[ModelTracker]:
        """
        Additional training for the best models.
        """
        if self.with_deep_train:
            deep_trainer = DeepTrainer(self.config_reader.get_deep_train_config(), model_list,
                                        DispatcherManager(JobType.DEEP.value, self.config.model_name,
                                                          self.config.cluster))
            deep_trainer.run()
            return deep_trainer.collect()

        return model_list

    def prepare_lammps(self, model_list: list[PotModel]) -> list[PotModel]:
        if self.with_conversion:
            for model in model_list:
                model.lampify()
                model.create_potential()

        return model_list

    def inference_bench(self, models: list[PotModel]):
        if self.with_inference:
            bencher = InferenceBencher(self.config_reader.get_bench_config(), models,
                              DispatcherManager(JobType.INF.value, self.config.model_name,
                                                self.config.cluster))
            bencher.run()

    def properties_simulation(self, models: list[PotModel]):
        if self.with_data_analysis:
            simulator = PropertiesSimulator(self.config_reader.get_prop_config(), models,
                              DispatcherManager(JobType.SIM.value, self.config.model_name,
                                                self.config.cluster))
            simulator.run()

    def get_model_out_paths(self) -> list[Path]:
        """
        Get the paths to the models.
        """
        iter_dirs: list[Path] = [d for d in self.config.sweep_path.iterdir() if d.is_dir()]
        subiter_dirs : list[Path] = [d for d in iter_dirs for d in d.iterdir() if d.is_dir()]
        out_paths: list[Path] = []
        for path in subiter_dirs:
            if (path / DEEP_TRAIN_DIR_NAME).exists():
                out_paths.append(path / DEEP_TRAIN_DIR_NAME)
            elif (path / OPTIM_DIR_NAME).exists():
                out_paths.append(path / OPTIM_DIR_NAME)
            else:
                raise ValueError(f"Could not find the model in {path}")
        return out_paths
