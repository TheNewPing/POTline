"""
Custom optimizer class for XPOT, based on the skopt.Optimizer class.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import yaml
from skopt import Optimizer # type: ignore
import xpot.loaders as load # type: ignore

from ..config_reader import HyperConfig
from ..model import create_model, CONFIG_NAME
from ..loss_logger import LossLogger, ModelTracker
from ..dispatcher import DispatcherManager

OPTIM_DIR_NAME: str = "hyper_search"

class PotOptimizer():
    """
    Custom optimizer class for XPOT, based on the skopt.Optimizer class.

    Args:
        - config: configuration data.
        - dispatcher_manager: manager for dispatching training
    """
    def __init__(
        self,
        config: HyperConfig,
        dispatcher_manager: DispatcherManager,
    ):
        self._config = config
        self._dispatcher_manager = dispatcher_manager
        self._iteration = 0
        self._subiter = 0
        self._iter_path = self._config.sweep_path / str(self._iteration) / str(self._subiter) / OPTIM_DIR_NAME
        self._fitted_models: list[ModelTracker] = []

        self._mlp_total = load.merge_hypers({}, self._config.optimizer_params)
        load.validate_hypers(self._mlp_total, self._config.optimizer_params)
        self._optimizable_params = load.get_optimisable_params(self._mlp_total)
        self._optimizer: Optimizer = Optimizer(
            dimensions=list(self._optimizable_params.values()),
            random_state=42,
            n_initial_points=self._config.n_initial_points,
        )

        self._loss_logger = LossLogger(self._config.sweep_path, self._get_keys())

    def run(self) -> list[ModelTracker]:
        """
        Run the optimisation sweep.
        """
        for _ in range(self._config.max_iter):
            self._fitted_models += self._optimize()
        self._loss_logger.tabulate_final_results()
        return self._fitted_models

    def _optimize(self) -> list[ModelTracker]:
        """
        Function for running optimisation sweep.
        """
        self._iteration += 1
        # Get parameters sets to evaluate
        next_params_list: list[dict] = self._ask()
        fit_trackers: list[ModelTracker] = []
        # Initialize all the models
        for next_params in next_params_list:
            self._subiter += 1
            self._iter_path = \
                self._config.sweep_path / str(self._iteration) / str(self._subiter) / OPTIM_DIR_NAME
            config_path: Path = self._prep_fit(next_params)
            fit_trackers.append(ModelTracker(
                create_model(self._config.model_name, config_path, self._iter_path),
                self._iteration, self._subiter, next_params))

        # Start the fitting process
        fit_cmd: str = fit_trackers[0].model.get_fit_cmd(deep=False)
        self._dispatcher_manager.set_job([fit_cmd],
                                               self._config.sweep_path,
                                               list(range(1,self._subiter+1)))
        self._dispatcher_manager.dispatch_job()

        # Collect the loss values
        self._dispatcher_manager.wait_job()
        for fit_tr in fit_trackers:
            fit_tr.valid_losses = fit_tr.model.collect_loss()
            self._loss_logger.write_error_file(fit_tr)

        # Tell the optimizer the results
        self._tell([fit_tr.params for fit_tr in fit_trackers],
                  [fit_tr.get_total_valid_loss(self._config.energy_weight) for fit_tr in fit_trackers])

        # Write the results to the parameters.csv file
        for fit_tr in fit_trackers:
            i: int = self._config.n_points - fit_tr.subiter + 1
            loss: float = fit_tr.get_total_valid_loss(self._config.energy_weight)
            key_values: list[str] = [str(i) for i in self._optimizer.Xi[-i]]
            self._loss_logger.write_param_result(fit_tr.iteration, fit_tr.subiter, loss, key_values)

        self._subiter = 0
        return fit_trackers

    def _get_keys(self) -> list[str]:
        """
        Get the keys of the optimizable parameters.
        """
        keys = [" ".join(i[0]) for i in self._optimizable_params] # type: ignore
        return list(map(str, keys))

    def _prep_fit(
        self,
        opt_values: dict,
    ) -> Path:
        """
        Prepare hyperparameters for the model fitting.

        Args:
            - opt_values: dictionary of hyperparameters.

        Returns:
            Path: path to the configuration file.
        """
        self._iter_path.mkdir(parents=True, exist_ok=True)

        self._mlp_total = load.reconstitute_lists(self._mlp_total, opt_values)
        self._mlp_total = load.prep_dict_for_dump(self._mlp_total)
        self._mlp_total = load.trim_empty_values(self._mlp_total)  # type: ignore
        self._mlp_total = load.convert_numpy_types(self._mlp_total)

        out_filepath: Path = self._iter_path / CONFIG_NAME
        with out_filepath.open("w+", encoding='utf-8') as f:
            yaml.safe_dump(dict(self._mlp_total), f)
        return out_filepath

    def _ask(self) -> list[dict]:
        """
        Ask the optimizer for a new set of parameters based on the current
        results of the optimisation.

        Returns:
            list: list of dictionaries of parameter to test.
        """
        param_values_list: list[list] = \
            self._optimizer.ask(self._config.n_points, self._config.strategy)
        return [dict(zip(self._optimizable_params.keys(), param_values))
                for param_values in param_values_list]

    def _tell(self, params_list: list[dict], results_list: list[float]):
        """
        Tell the optimizer the result of the last iteration, as well as the
        parameter values used to achieve it.

        Args:
            - params_list: list of dictionaries of parameter tested.
            - results_list: list of results from the last iteration.
        """
        # 1. make sure that we get the order correct
        locations_list = [[params[name] for name in self._optimizable_params] for params in params_list]
        # 2. tell the optimizer
        self._optimizer.tell(locations_list, results_list)

    def dump_optimizer(self, filename: str = 'optimizer.pkl'):
        """
        Dump the optimizer to a file.
        """
        filepath: Path = self._config.sweep_path / filename
        with filepath.open("wb") as f:
            pickle.dump(self._optimizer, f)

    def load_optimizer(self, filename: str = 'optimizer.pkl'):
        """
        Load the optimizer from a file.
        """
        filepath: Path = self._config.sweep_path / filename
        with filepath.open("rb") as f:
            self._optimizer = pickle.load(f)

        if len(self._optimizer.space.dimensions) != len(self._optimizable_params):
            raise ValueError(
                "The optimizer and the optimisable parameters "
                "have different lengths. The optimizer cannot be "
                "loaded."
            )
