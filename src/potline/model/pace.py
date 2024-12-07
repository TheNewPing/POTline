"""
Pacemaker wrapper for fitting ACE potentials using XPOT in HPC.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xpot import maths # type: ignore

from .model import PotModel
from ..dispatcher import create_dispatcher

PACE_DEFAULTS_PATH: str = str(Path(__file__).parent / "defaults" / "ace_defaults.json")
PACE_CONFIG_NAME: str = "xpot-ace.yaml"

class PotPACE(PotModel):
    """
    PACE implementation.
    Requires pacemaker.
    """
    def __init__(self, config_filepath: Path,
                 out_path: Path,
                 energy_weight: float,
                 hpc: bool,
                 default_filepath: Path = PACE_DEFAULTS_PATH):
        super().__init__(config_filepath, out_path, energy_weight, hpc, default_filepath)

    def dispatch_fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        subiter: int,
    ):
        self._prep_fit(opt_values, iteration, subiter)
        input_file = self.iter_path / PACE_CONFIG_NAME
        self._write_input_file(input_file)

        if self.hpc:
            fit_job_options: dict = {
                'snellius': {
                    'conda': True,
                    'env_pace': True,
                    'cuda': True,
                },
                'job_name': "fit_pace",
                'output': f"{self.iter_path}/fit_%j.out",
                'error': f"{self.iter_path}/fit_%j.err",
                'time': "12:00:00",
                'mem': "50G",
                'partition': "gpu",
                'nodes': 1,
                'ntasks': 1,
                'cpus_per_task': 16,
                'gpus': 1,
            }
            self.dispatcher = create_dispatcher(f"pacemaker {input_file}", fit_job_options)
            self.dispatcher.dispatch()

    def collect_loss(self, wait_id: int, iteration: int, subiter: int) -> float:
        """
        Collect the loss from the fitting process.

        Parameters
        ----------
        wait_id : int
            The job ID to wait for.
        iteration : int
            The iteration number.
        subiter : int

        Returns
        -------
        float
            The loss value.
        """
        self.iteration = iteration
        self.subiter = subiter
        self.iter_path = os.path.join(self.out_path, str(self.iteration), str(self.subiter))

        wait_job = Slurm()
        while True:
            wait_job.squeue.update_squeue()
            if wait_id not in wait_job.squeue.jobs:
                break
            time.sleep(10)

        os.chdir(self.iter_path)

        tmp_train = self._calculate_errors(validation=False)
        tmp_test = self._calculate_errors(validation=True)

        train_errors = self._validate_errors(tmp_train, maths.get_rmse, [1, 0.5])
        test_errors = self._validate_errors(tmp_test, maths.get_rmse, [1, 0.5])

        self._process_errors(
            train_errors[0], test_errors[0], "atomistic_errors.csv"
        )

        loss = self._process_errors(
            train_errors[1], test_errors[1], "loss_function_errors.csv"
        )

        return loss


    def _write_input_file(
        self,
        out_filepath: Path = PACE_CONFIG_NAME):
        """
        Write the input file for the ACE potential from the hyperparameter +
        dictionary.

        Parameters
        ----------
        filename : str
            Path to the input file.
        """
        with out_filepath.open("w+", encoding='utf-8') as f:
            yaml.safe_dump(dict(self.mlp_total), f)

    def _collect_raw_errors(self, errors_filepath: Path) -> pd.DataFrame:
        """
        Collect errors from the fitting process.

        Parameters
        ----------
        filename : str
            The file to read errors from.

        Returns
        -------
        pd.DataFrame
            The dataframe of the errors from the fitting process.

        """
        df = pd.read_pickle(errors_filepath, compression="gzip")
        return df

    def _calculate_errors(
        self,
        validation: bool = False,
    ) -> dict[str, list[float]]:
        """
        Validate the potential from pickle files produced by :code:`pacemaker`
        during the fitting process.

        Parameters
        ----------
        validation : bool
            If True, calculate validation errors, otherwise calculate training
            errors.

        Returns
        -------
        dict
            The errors as a dictionary of lists.
        """
        if validation:
            errors = self._collect_raw_errors("test_pred.pckl.gzip")
        else:
            errors = self._collect_raw_errors("train_pred.pckl.gzip")

        n_per_structure = errors["NUMBER_OF_ATOMS"].values.tolist()

        ref_energy = errors["energy_corrected"].values.tolist()
        pred_energy = errors["energy_pred"].values.tolist()

        energy_diff = [pred - ref for pred, ref in zip(pred_energy, ref_energy)]

        ref_forces = np.concatenate(errors["forces"].to_numpy(), axis=None)
        pred_forces = np.concatenate(
            errors["forces_pred"].to_numpy(), axis=None
        )
        forces_diff = [pred - ref for pred, ref in zip(pred_forces, ref_forces)]

        out_errors = {
            "energies": energy_diff,
            "forces": forces_diff,
            "atom_counts": n_per_structure,
        }

        return out_errors

