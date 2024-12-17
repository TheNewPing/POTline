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
        - model_tracker: model to train
        - dispatcher_manager: manager for dispatching training jobs
    """
    def __init__(self, config: DeepTrainConfig, tracker_list: list[ModelTracker],
                 dispatcher_manager: DispatcherManager):
        self._config = config
        self._tracker_list = tracker_list
        self._dispatcher_manager = dispatcher_manager
        self._fit_cmd = tracker_list[0].model.get_fit_cmd(deep=True)
        for tracker in self._tracker_list:
            self._iter_path = \
                self._config.sweep_path / str(tracker.iteration) / str(tracker.subiter)
            self._out_path = self._iter_path / DEEP_TRAIN_DIR_NAME
            self._out_path.mkdir(exist_ok=True)
            tracker.model.switch_out_path(self._out_path)
            tracker.model.set_config_maxiter(self._config.max_epochs)

        self._loss_logger = LossLogger(self._out_path)

    def run(self):
        cmds: list[str] = ['declare -A path_map']
        # prepare path mapping cmds
        for i, tracker in enumerate(self._tracker_list):
            cmds.append(f'path_map[{i+1}]="{str(tracker.model.get_out_path())}"')
        cmds.append('index=$SLURM_ARRAY_TASK_ID')
        # TODO: refactor directories structure???? (invert job type and iteration)
        self._dispatcher_manager.set_job(cmds, p, ids, n_cpu=1)
        self._tracker_list.model.get_fit_cmd(self._dispatcher_manager, deep=True)

    def collect(self) -> ModelTracker:
        self._tracker_list.valid_losses = self._tracker_list.model.collect_loss()
        self._loss_logger.write_error_file(self._tracker_list)
        return self._tracker_list
