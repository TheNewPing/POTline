"""
DeepTrainer class for training after hyperparameter search.
"""

from ..config_reader import DeepTrainConfig
from ..dispatcher import DispatcherFactory
from ..loss_logger import LossLogger, ModelTracker

DEEP_TRAIN_DIR_NAME: str = 'deep_train'

class DeepTrainer():
    """
    Abstract class for training after hyperparameter search.
    """
    def __init__(self, config: DeepTrainConfig, model_tracker: ModelTracker,
                 dispatcher_factory: DispatcherFactory):
        self._config = config
        self._model_tracker = model_tracker
        self._iter_path = \
            self._config.sweep_path / str(self._model_tracker.iteration) / str(self._model_tracker.subiter)
        self._pot_path = self._model_tracker.model.get_last_pot_path()
        self._out_path = self._iter_path / DEEP_TRAIN_DIR_NAME
        self._out_path.mkdir(exist_ok=True)
        self._model_tracker.model.switch_out_path(self._out_path)
        self._model_tracker.model.set_config_maxiter(self._config.max_epochs)
        self._dispatcher_factory = dispatcher_factory

        self._loss_logger = LossLogger(self._out_path)

    def run(self):
        self._model_tracker.model.dispatch_fit(self._dispatcher_factory, ['-p', str(self._pot_path)])

    def collect(self) -> ModelTracker:
        self._model_tracker.train_losses = self._model_tracker.model.collect_loss(validation=False)
        self._model_tracker.test_losses = self._model_tracker.model.collect_loss(validation=True)
        self._loss_logger.write_error_file(self._model_tracker)
        return self._model_tracker
