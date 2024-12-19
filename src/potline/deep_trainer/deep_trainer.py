"""
DeepTrainer class for training after hyperparameter search.
"""

from ..config_reader import DeepTrainConfig
from ..dispatcher import DispatcherManager
from ..loss_logger import LossLogger, ModelTracker

DEEP_TRAIN_DIR_NAME: str = 'deep_train'

class DeepTrainer():
    """
    Class for training after hyperparameter search.

    Args:
        - config: configuration for deep training
        - tracker_list: models to train
        - dispatcher_manager: manager for dispatching training jobs
    """
    def __init__(self, config: DeepTrainConfig, tracker_list: list[ModelTracker],
                 dispatcher_manager: DispatcherManager):
        self._config = config
        self._tracker_list = tracker_list
        self._dispatcher_manager = dispatcher_manager
        self._fit_cmd = tracker_list[0].model.get_fit_cmd(deep=True)
        self._out_path = self._config.sweep_path / DEEP_TRAIN_DIR_NAME
        self._out_path.mkdir(exist_ok=True)
        for i, tracker in enumerate(self._tracker_list):
            iter_path = self._out_path / str(i)
            iter_path.mkdir(exist_ok=True)
            tracker.model.switch_out_path(iter_path)
            tracker.model.set_config_maxiter(self._config.max_epochs)

        self._loss_logger = LossLogger(self._out_path)

    def run(self):
        self._dispatcher_manager.set_job([self._fit_cmd],
                                         self._out_path,
                                         self._config.job_config,
                                         list(range(1,len(self._tracker_list)+1)))
        self._dispatcher_manager.dispatch_job()

    def collect(self) -> list[ModelTracker]:
        self._dispatcher_manager.wait_job()
        for tracker in self._tracker_list:
            tracker.valid_losses = tracker.model.collect_loss()
            self._loss_logger.write_error_file(tracker)
        return self._tracker_list
