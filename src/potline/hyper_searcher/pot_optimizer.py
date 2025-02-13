"""
Custom optimizer class for XPOT, based on the skopt.Optimizer class.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import math

import yaml
from skopt import Optimizer # type: ignore
import xpot.loaders as load # type: ignore

from ..config_reader import ConfigReader
from ..model import create_model, CONFIG_NAME, Losses
from ..loss_logger import LossLogger, ModelTracker

OPTIM_DIR_NAME: str = "hyper_search"

class PotOptimizer():
    """
    Custom optimizer class for XPOT, based on the skopt.Optimizer class.

    Args:
        - config_path: path to the configuration file.
        - restart_optimizer: whether to restart the optimizer.
        - iteration: iteration number.
    """
    def __init__(self, config_path: Path, restart_optimizer: bool = False, iteration: int = 1):
        self._config = ConfigReader(config_path).get_optimizer_config()
        self._config_path = config_path
        self._restart_optimizer = restart_optimizer
        self._iteration = iteration
        self._subiter = 1
        self._out_path = self._config.sweep_path / OPTIM_DIR_NAME
        self._iter_path = self._out_path / str(self._iteration) / str(self._subiter)

        self._mlp_total = load.merge_hypers({}, self._config.optimizer_params)
        load.validate_hypers(self._mlp_total, self._config.optimizer_params)
        self._optimizable_params = load.get_optimisable_params(self._mlp_total)
        if self._iteration == 1:
            # Create a new optimizer only if it is the first iteration
            print("Creating optimizer...")
            self._out_path.mkdir(parents=True, exist_ok=True)
            self._optimizer: Optimizer = Optimizer(
                dimensions=list(self._optimizable_params.values()),
                random_state=42,
                n_initial_points=self._config.n_initial_points,
            )
        else:
            print("Loading optimizer...")
            self.load_optimizer()

        self._loss_logger = LossLogger(self._out_path, self._get_keys(), no_init=(self._iteration != 1))

    def run(self) -> None:
        """
        Function for running optimisation sweep.
        """

        if self._restart_optimizer:
            self._collect_losses()

        if self._iteration <= self._config.max_iter:
            self._setup_trackers()
        else:
            # Tabulate the final results
            self._loss_logger.tabulate_final_results()
            self.dump_optimizer()
            print("Optimization completed.")

    def _setup_trackers(self) -> list[ModelTracker]:
        # Get parameters sets to evaluate
        next_params_list: list[dict] = self._ask()
        fit_trackers: list[ModelTracker] = []
        # Initialize all the models
        for next_params in next_params_list:
            self._iter_path = self._out_path / str(self._iteration) / str(self._subiter)
            self._prep_fit(next_params)
            new_tracker = ModelTracker(
                create_model(self._config.model_name, self._iter_path),
                self._iteration, self._subiter, next_params)
            new_tracker.save_info(self._iter_path)
            fit_trackers.append(new_tracker)
            self._subiter += 1
        self.dump_optimizer()
        return fit_trackers

    def _collect_losses(self) -> None:
        # Get the model trackers
        fit_trackers: list[ModelTracker] = PotOptimizer.get_model_trackers(self._config.sweep_path,
                                                                           self._config.model_name)
        # Filter models with the previous iteration
        fit_trackers = [fit_tr for fit_tr in fit_trackers if fit_tr.iteration == self._iteration-1]

        # Collect the loss values
        for fit_tr in fit_trackers:
            try:
                fit_tr.valid_losses = fit_tr.model.collect_loss()
            except Exception as e:
                print(f"Error collecting [{fit_tr.iteration};{fit_tr.subiter}]")
                print(e)
                if self._config.handle_collect_errors:
                    fit_tr.valid_losses = Losses(math.nan, math.nan)
                else:
                    print('Dumping optimizer...')
                    self.dump_optimizer()
                    raise e
            finally:
                self._loss_logger.write_error_file(fit_tr)
                fit_tr.save_info(fit_tr.model.get_out_path())

        # Tell the optimizer the results
        self._tell([fit_tr.params for fit_tr in fit_trackers],
                  [fit_tr.get_total_valid_loss(self._config.energy_weight) for fit_tr in fit_trackers])

        # Write the results to the parameters.csv file
        for fit_tr in fit_trackers:
            loss: float = fit_tr.get_total_valid_loss(self._config.energy_weight)
            key_values: list[str] = [str(i) for i in
                                    [fit_tr.params[name] for name in self._optimizable_params]]
            self._loss_logger.write_param_result(fit_tr.iteration, fit_tr.subiter, loss, key_values)

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
        filepath: Path = self._out_path / filename
        with filepath.open("wb") as f:
            pickle.dump(self._optimizer, f)

    def load_optimizer(self, filename: str = 'optimizer.pkl'):
        """
        Load the optimizer from a file.
        """
        filepath: Path = self._out_path / filename
        with filepath.open("rb") as f:
            self._optimizer = pickle.load(f)

        if len(self._optimizer.space.dimensions) != len(self._optimizable_params):
            raise ValueError(
                "The optimizer and the optimisable parameters "
                "have different lengths. The optimizer cannot be "
                "loaded."
            )

    @staticmethod
    def get_model_trackers(sweep_path: Path, model_name: str) -> list[ModelTracker]:
        """
        Get the model trackers from the sweep path.

        Args:
            - sweep_path: path to the sweep
            - model_name: name of the model

        Returns:
            - list of model trackers from the hyperparameter search directory
        """
        hyp_path: Path = sweep_path / OPTIM_DIR_NAME
        iter_dirs: list[Path] = [d for d in hyp_path.iterdir() if d.is_dir()]
        model_dirs: list[Path] = [d for d in iter_dirs for d in d.iterdir() if d.is_dir()]
        models: list[ModelTracker] = []
        for model_path in model_dirs:
            if model_path.is_dir():
                model_tracker = ModelTracker.from_path(model_name, model_path)
                models.append(model_tracker)
        return models
