"""
Loss logger
"""

import csv
from pathlib import Path
import pickle

from tabulate import tabulate
from xpot import maths # type: ignore
import yaml

from ..model import PotModel, Losses, create_model

ERROR_FILENAME = "loss_function_errors.csv"
ERROR_PARAMETER_FILENAME = "parameters.csv"
INFO_FILENAME = "model_info.yaml"
INFO_PARM_FILENAME = "model_params.pckl"

class ModelTracker():
    """
    Class to track the progress of a job in the optimisation sweep.

    Args:
        - model: model to track
        - iteration: iteration number
        - subiter: subiteration number
        - params: parameters of the model
        - valid_losses: valid losses of the model
    """
    def __init__(self, model: PotModel, iteration: int, subiter: int,
                 params: dict, valid_losses: Losses | None = None) -> None:
        self.model = model
        self.iteration = iteration
        self.subiter = subiter
        self.params = params
        self.valid_losses = valid_losses

    def get_total_valid_loss(self, energy_weight: float) -> float:
        """
        Get the total valid loss from the model.

        Args:
            - energy_weight: weight of the energy loss
        """
        if self.valid_losses is None:
            raise ValueError("valid loss not calculated.")
        return maths.calculate_loss(self.valid_losses.energy, self.valid_losses.force, energy_weight)

    def save_info(self, out_path: Path):
        """
        Save the model information to a file.

        Args:
            - out_path: path to save the information
        """
        with (out_path / INFO_FILENAME).open("w", encoding='utf-8') as f:
            loss = {
                'valid_energy_loss': self.valid_losses.energy,
                'valid_force_loss': self.valid_losses.force
            } if self.valid_losses is not None else {}
            data = {
                'iteration': self.iteration,
                'subiteration': self.subiter,
                **loss,
            }
            yaml.dump(data, f)

        with (out_path / INFO_PARM_FILENAME).open("wb") as f:
            pickle.dump(self.params, f)

    @staticmethod
    def from_path(model_name: str, model_path: Path) -> 'ModelTracker':
        """
        Create a model tracker from a path.

        Args:
            - model_name: name of the model
            - model_path: path to the model, used to recover the model

        Returns:
            ModelTracker: the model tracker
        """
        model = create_model(model_name, model_path)
        with (model_path / INFO_FILENAME).open("r", encoding='utf-8') as f:
            data: dict = yaml.safe_load(f)
            iteration = int(data['iteration'])
            subiter = int(data['subiteration'])
            energy_loss: str | None = data.get('valid_energy_loss')
            force_loss: str | None = data.get('valid_force_loss')
            valid_losses = Losses(float(data['valid_energy_loss']), float(data['valid_force_loss'])) \
                if energy_loss and force_loss else None
        with (model_path / INFO_PARM_FILENAME).open("rb") as f:
            params = pickle.load(f)

        return ModelTracker(model, iteration, subiter, params, valid_losses)

class LossLogger():
    """
    Loss logger

    Args:
        - sweep_path: path to the sweep
        - keys: keys of the optimized parameters
    """
    def __init__(self, sweep_path: Path, keys: list[str] | None = None, no_init: bool = False):
        self._sweep_path = sweep_path
        self._error_filepath = sweep_path / ERROR_FILENAME
        self._param_filepath = sweep_path / ERROR_PARAMETER_FILENAME
        self._keys = keys
        if not no_init:
            self._initialise_csvs()

    def tabulate_final_results(self):
        """
        Tabulate the final results of the optimisation into pretty tables.
        """
        def tabulate_csv(filepath: Path):
            with filepath.open(encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)
                table = tabulate(rows, headers="firstrow", tablefmt="github")
            out_path = filepath.parent / filepath.stem
            with out_path.open("a+", encoding='utf-8') as f:
                f.write(table)

        tabulate_csv(self._error_filepath)
        tabulate_csv(self._param_filepath)

    def write_error_file(self, job_tracker: ModelTracker):
        """
        Write the error values to a file.

        Args:
            - job_tracker: the job tracker to write to the file
        """
        if job_tracker.valid_losses is None:
            raise ValueError("Losses not calculated.")
        output_data = [job_tracker.iteration, job_tracker.subiter,
                       job_tracker.valid_losses.energy,
                       job_tracker.valid_losses.force]
        with self._error_filepath.open("a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(output_data)

    def _initialise_csvs(self):
        """
        Initialise the CSV files for the optimisation.
        """
        print("Initialising CSV files...")
        if self._keys:
            with self._param_filepath.open("w+", encoding='utf-8') as f:
                f.write("iteration,subiteration,loss," + ",".join(self._keys) + "\n")
        with self._error_filepath.open("w+", encoding='utf-8') as f:
            f.write(
                "Iteration,"
                + "Subiteration,"
                + "valid Δ Energy,"
                + "valid Δ Force"
                + "\n"
            )

    def write_param_result(
        self,
        iteration: int,
        subiteration: int,
        loss: float,
        key_values: list[str]
    ):
        """
        Write the loss to the parameters.csv file.

        Args:
            - iteration: iteration number
            - subiteration: subiteration number
            - loss: loss value
            - key_values: optimizable parameter values
        """
        if self._keys is None:
            raise ValueError("Keys must be provided to write to the parameters file.")
        with self._param_filepath.open("a", encoding='utf-8') as f:
            f.write(
                f"{iteration},"
                + f"{subiteration},"
                + f"{loss},"
                + ",".join(key_values)
                + "\n"
            )
