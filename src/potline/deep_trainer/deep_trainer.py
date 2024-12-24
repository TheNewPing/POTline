"""
DeepTrainer class for training after hyperparameter search.
"""

from pathlib import Path

from ..config_reader import ConfigReader
from ..loss_logger import LossLogger, ModelTracker

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
