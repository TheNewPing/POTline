"""
LAMMPS experiment module.
"""

from pathlib import Path
import shutil

from ..config_reader import ConfigReader
from ..dispatcher import DispatcherManager
from ..loss_logger import ModelTracker
from ..config_reader import JobConfig

class Experiment():
    """
    Class for running the LAMMPS experiments.
    """
    @staticmethod
    def prep_exp(out_path: Path, copy_dir: Path, tracker_list: list[ModelTracker]) -> None:
        """
        Prepare the experiment directories.

        Args:
            - out_path: the path to the output directory.
            - copy_dir: the path to the directory to copy, it should contain the experiment scripts.
            - tracker_list: the list of model trackers to use in the experiments.
        """
        out_path.mkdir(exist_ok=True)

        for i, tracker in enumerate(tracker_list):
            iter_path = out_path / str(i+1)
            iter_path.mkdir(exist_ok=True)
            shutil.copy(tracker.model.get_pot_path(), iter_path)
            tracker.save_info(iter_path)
            for file in copy_dir.iterdir():
                if file.is_file():
                    shutil.copy(file, iter_path)

    @staticmethod
    def run_exp(config_path: Path, out_path: Path, copy_dir: Path, exp_name: str, command: str,
                n_models: int, job_config: JobConfig, job_type_prep: str, job_type_run: str,
                model: str, dependency: int | None = None,) -> int:
        """
        Run the experiment.

        Args:
            - config_path: the path to the configuration file.
            - out_path: the path to the output directory.
            - copy_dir: the path to the directory to copy, it should contain the experiment scripts.
            - exp_name: the name of the experiment.
            - command: the command to run.
            - n_models: the number of models to run.
            - job_config: the job configuration.
            - job_type_prep: the job type for the preparation.
            - job_type_run: the job type for the run.
            - model: the model name.
            - dependency: the job dependency.

        Returns:
            int: The id of experiments jobs.
        """
        out_path.mkdir(exist_ok=True)
        gen_config = ConfigReader(config_path).get_general_config()

        prep_manager = DispatcherManager(job_type_prep, model, job_config.cluster)
        run_manager = DispatcherManager(job_type_run, model, job_config.cluster)
        cli_path: Path = gen_config.repo_path / 'src' / 'run_exp.py'

        # init job
        init_cmd: str = f'{gen_config.python_bin} {cli_path}' + \
                        f' --config {config_path}' + \
                        f' --copydir {copy_dir}' + \
                        f' --expname {exp_name}'
        prep_manager.set_job([init_cmd], out_path, job_config, dependency=dependency)
        init_id = prep_manager.dispatch_job()

        # run jobs
        cpus_per_task = int(job_config.slurm_opts['cpus_per_task'])
        ntasks = int(job_config.slurm_opts['ntasks'])
        run_manager.set_job([command + f' {cpus_per_task} {ntasks}'], out_path, job_config,
                            dependency=init_id, array_ids=list(range(1, n_models+1)))
        return run_manager.dispatch_job()
