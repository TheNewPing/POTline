"""
DeepTrainer class for training after hyperparameter search.
"""

from pathlib import Path

from ..config_reader import ConfigReader
from ..loss_logger import LossLogger, ModelTracker
from ..dispatcher import DispatcherManager, JobType
from ..model import get_fit_cmd

DEEP_TRAIN_DIR_NAME: str = 'deep_train'

class DeepTrainer():
    """
    Class for training after hyperparameter search.

    Args:
        - config: configuration for deep training
        - tracker_list: models to train
    """
    def __init__(self, config_path: Path, tracker_list: list[ModelTracker]):
        self._config = ConfigReader(config_path).get_deep_train_config()
        self._config_path = config_path
        self._tracker_list = tracker_list
        self._out_path = self._config.sweep_path / DEEP_TRAIN_DIR_NAME

    def prep_deep(self) -> None:
        self._out_path.mkdir(exist_ok=True)
        for i, tracker in enumerate(self._tracker_list):
            iter_path = self._out_path / str(i+1)
            iter_path.mkdir(exist_ok=True)
            tracker.model.switch_out_path(iter_path)
            tracker.model.set_config_maxiter(self._config.max_epochs)
            tracker.save_info(iter_path)

    def collect(self):
        loss_logger = LossLogger(self._out_path)
        for tracker in self._tracker_list:
            tracker.valid_losses = tracker.model.collect_loss()
            loss_logger.write_error_file(tracker)
            tracker.save_info(tracker.model.get_out_path())

    @staticmethod
    def get_model_trackers(sweep_path: Path, model_name: str) -> list[ModelTracker]:
        """
        Get the model trackers from the sweep path.

        Args:
            - sweep_path: path to the sweep
            - model_name: name of the model

        Returns:
            - list of model trackers from the deep train directory
        """
        deep_path: Path = sweep_path / DEEP_TRAIN_DIR_NAME
        model_dirs: list[Path] = [d for d in deep_path.iterdir() if d.is_dir()]
        print(f"Found {len(model_dirs)} models in {deep_path}")
        print(f"{model_dirs}")
        models: list[ModelTracker] = []
        for model_path in model_dirs:
            if model_path.is_dir():
                model_tracker = ModelTracker.from_path(model_name, model_path)
                models.append(model_tracker)
        return models

    @staticmethod
    def run_deep(config_path: Path, dependency: int | None = None) -> int:
        """
        Run deep training.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        deep_config = ConfigReader(config_path).get_deep_train_config()
        gen_config = ConfigReader(config_path).get_general_config()
        cli_path: Path = gen_config.repo_path/ 'src' / 'run_deep.py'
        out_path: Path = deep_config.sweep_path / DEEP_TRAIN_DIR_NAME
        out_path.mkdir(exist_ok=True)
        deep_manager = DispatcherManager(
            JobType.DEEP.value, deep_config.model_name, deep_config.job_config.cluster)
        watch_manager = DispatcherManager(
            JobType.WATCH_DEEP.value, deep_config.model_name, deep_config.job_config.cluster)

        # init job
        init_cmd: str = f'{gen_config.python_bin} {cli_path} --config {config_path}'
        watch_manager.set_job([init_cmd], out_path, deep_config.job_config, dependency=dependency)
        init_id = watch_manager.dispatch_job()

        # fit jobs
        deep_cmd: str = get_fit_cmd(deep_config.model_name, deep=True)
        deep_manager.set_job([deep_cmd], out_path, deep_config.job_config, dependency=init_id,
                            array_ids=list(range(1, deep_config.best_n_models+1)))
        fit_id = deep_manager.dispatch_job()

        # collect job
        coll_cmd: str = f'{gen_config.python_bin} {cli_path} --config {config_path} --collect'
        watch_manager.set_job([coll_cmd], out_path, deep_config.job_config, dependency=fit_id)
        return watch_manager.dispatch_job()
