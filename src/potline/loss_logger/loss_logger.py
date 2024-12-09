"""
Loss logger
"""

import csv
from pathlib import Path

from tabulate import tabulate

from ..model import ModelTracker

ERROR_FILENAME = "loss_function_errors.csv"
PARAMETER_FILENAME = "parameters.csv"

class LossLogger():
    """
    Loss logger
    """
    def __init__(self, sweep_path: Path, keys: list[str] | None = None):
        self._sweep_path = sweep_path
        self._error_filepath = sweep_path / ERROR_FILENAME
        self._param_filepath = sweep_path / PARAMETER_FILENAME
        self._keys = keys
        self._initialise_csvs()

    def tabulate_final_results(self):
        """
        Tabulate the final results of the optimisation into pretty tables, with
        filenames
        """
        def tabulate_csv(filepath: Path):
            with filepath.open(encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)
                table = tabulate(rows, headers="firstrow", tablefmt="github")
            with filepath.open("a+", encoding='utf-8') as f:
                f.write(table)

        tabulate_csv(self._error_filepath)
        tabulate_csv(self._param_filepath)

    def write_error_file(self, job_tracker: ModelTracker):
        """
        Write the error values to a file.
        """
        if job_tracker.test_losses is None or job_tracker.train_losses is None:
            raise ValueError("Losses not calculated.")
        output_data = [job_tracker.iteration, job_tracker.subiter,
                        job_tracker.train_losses.energy, job_tracker.test_losses.energy,
                        job_tracker.train_losses.force, job_tracker.test_losses.force]
        with self._error_filepath.open("a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(output_data)

    def _initialise_csvs(self):
        """
        Initialise the CSV files for the optimisation.
        """
        if self._keys:
            with self._param_filepath.open("w+", encoding='utf-8') as f:
                f.write("iteration,subiteration,loss," + ",".join(self._keys) + "\n")
        with self._error_filepath.open("w+", encoding='utf-8') as f:
            f.write(
                "Iteration,"
                + "Subiteration,"
                + "Train Δ Energy,"
                + "Test Δ Energy,"
                + "Train Δ Force,"
                + "Test Δ Force"
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

    # TODO: Fix this
    # def plot_results(self, path: str) -> None:
    #     """
    #     Function to create scikit-optimize results using inbuilt functions.

    #     Parameters
    #     ----------
    #     path : str
    #         Path of directory to save plots to.
    #     """
    #     data = self._optimizer.get_result()
    #     a = plots.plot_objective(data, levels=20, size=3)
    #     plt.tight_layout()
    #     a.figure.savefig(f"{path}/objective.pdf")

    #     b = plots.plot_evaluations(data)
    #     b.figure.savefig(f"{path}/evaluations.pdf")
