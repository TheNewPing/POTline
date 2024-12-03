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
from simple_slurm import Slurm # type: ignore

from .hpc_mlp import HPCMLP

_this_file = Path(__file__).resolve()
_exec_path = Path(os.getcwd()).resolve()

class HPCPACE(HPCMLP):
    """
    Class for optimizing ACE potentials using XPOT.
    Requires Slurm and Pacemaker.

    Parameters
    ----------
    infile: str
        Path to input .hjson file containing XPOT and ML parameters.
    """
    def __init__(self, infile: str) -> None:
        super().__init__(infile, "ace_defaults.json")

    def write_input_file(
        self,
        filename: str = "xpot-ace.yaml",
    ) -> None:
        """
        Write the input file for the ACE potential from the hyperparameter +
        dictionary.

        Parameters
        ----------
        filename : str
            Path to the input file.
        """
        with open(filename, "w+", encoding='utf-8') as f:
            yaml.safe_dump(dict(self.mlp_total), f)  # type: ignore

    def dispatch_fit(
        self,
        opt_values: dict[str, str | int | float],
        iteration: int,
        subiter: int,
        filename: str = "xpot-ace.yaml",
    ) -> int:
        """
        The main fitting function for creating an ACE potential. This function
        does the following:
        1. Replace old values with new hyperparameters from the optimiser.
        2. Reconstitute lists in the hyperparameter dictionary.
        3. Write the input file.
        4. Run pacemaker.
        5. Collect errors from the fitting process, write outputs, and return
        the loss value.

        Parameters
        ----------
        opt_values : dict
            The hyperparameters to be used in the fitting process.
        iteration : int
            The iteration number.
        subiter : int
            The subiteration number.
        filename : str
            Path/Name for the input file to be written to.

        Returns
        -------
        int
            The job ID of the fitting process.
        """
        self.prep_fit(opt_values, iteration, subiter)
        self.write_input_file(filename)

        fit_job = Slurm(
            job_name="fit_pace",
            output=f"{self.iter_path}/fit_%j.out",
            error=f"{self.iter_path}/fit_%j.err",
            time="48:00:00",
            mem="60G",
            partition="gpu",
            nodes=1,
            ntasks=1,
            cpus_per_task=16,
            gpus=1,
        )
        fit_job.add_cmd("module load 2024")
        fit_job.add_cmd("module load Miniconda3/24.7.1-0")
        fit_job.add_cmd("module load 2022")
        fit_job.add_cmd("module load cuDNN/8.4.1.50-CUDA-11.7.0")
        fit_job.add_cmd("export LD_LIBRARY_PATH=/home/erodaro/.conda/envs/pl/lib/:$LD_LIBRARY_PATH")
        fit_job.add_cmd("source $(conda info --base)/etc/profile.d/conda.sh")
        fit_job.add_cmd("conda activate pl")
        return fit_job.sbatch(f"pacemaker {filename}")

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
        self.iter_path = os.path.join(self.sweep_path, str(self.iteration), str(self.subiter))

        wait_job = Slurm()
        while True:
            wait_job.squeue.update_squeue()
            if wait_id not in wait_job.squeue.jobs:
                break
            time.sleep(10)

        os.chdir(self.iter_path)

        tmp_train = self.calculate_errors(validation=False)
        tmp_test = self.calculate_errors(validation=True)

        train_errors = self.validate_errors(tmp_train, maths.get_rmse, [1, 0.5])
        test_errors = self.validate_errors(tmp_test, maths.get_rmse, [1, 0.5])

        self.process_errors(
            train_errors[0], test_errors[0], "atomistic_errors.csv"
        )

        loss = self.process_errors(
            train_errors[1], test_errors[1], "loss_function_errors.csv"
        )

        return loss

    def collect_raw_errors(self, filename: str) -> pd.DataFrame:
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
        df = pd.read_pickle(filename, compression="gzip")
        return df

    def calculate_errors(
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
            errors = self.collect_raw_errors("test_pred.pckl.gzip")
        else:
            errors = self.collect_raw_errors("train_pred.pckl.gzip")

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
