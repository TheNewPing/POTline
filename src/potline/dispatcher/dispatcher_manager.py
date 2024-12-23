"""
dispatcher factory
"""

from pathlib import Path
import subprocess

from .slurm_preset import get_slurm_options
from .slurm_dispatcher import SlurmDispatcher
from ..config_reader import JobConfig

class DispatcherManager():
    """
    Dispatcher manager.

    Args:
        - job_type: type of job to run
        - model: model name
        - cluster: cluster to run the job on
    """
    def __init__(self,
                 job_type: str,
                 model: str,
                 cluster: str):
        self._job_type = job_type
        self._model = model
        self._cluster = cluster
        self._dispatcher: SlurmDispatcher | None = None

    def set_job(self, commands: list[str], out_path: Path,
                job_config: JobConfig,
                array_ids: list[int] | None = None,
                dependency: int | None = None,
                hold: bool = False):
        """
        Create a dispatcher based on the options.

        Args:
            - commands: commands to run
            - out_path: path to the output directory
            - job_config: job configuration
            - array_ids: array ids to run
            - dependency: job dependency
            - hold: whether to hold the job

        Returns:
            Dispatcher: the dispatcher to use.
        """
        is_array_job = array_ids is not None

        # Define slurm job requirements
        slurm_dict = job_config.slurm_watcher if not is_array_job else job_config.slurm_opts
        options = get_slurm_options(
            self._cluster, self._job_type, out_path, self._model,
            slurm_dict, array_ids, dependency)
        options.update({'hold': hold})

        # Setup environment
        source_cmds = [f'source {str(cmd)}' for cmd in job_config.modules]
        py_cmds = [f'python {str(script)}' for script in job_config.py_scripts] if is_array_job else []
        array_cmds = ['cd $SLURM_ARRAY_TASK_ID'] if array_ids else []
        export_cmds = ['export OMP_PROC_BIND=spread', 'export OMP_PLACES=threads']
        tot_cmds = export_cmds + array_cmds + source_cmds + py_cmds + commands

        # Create dispatcher
        self._dispatcher = SlurmDispatcher(tot_cmds, options)

    def dispatch_job(self) -> int:
        """
        Dispatch the job.
        """
        if self._dispatcher is None:
            raise ValueError("No job has been set yet.")
        return self._dispatcher.dispatch()

    def wait_job(self):
        """
        Wait for the job to finish.
        """
        if self._dispatcher is None:
            raise ValueError("No job has been set yet.")
        self._dispatcher.wait()

    @staticmethod
    def release_id(job_id: int, dependency: int | None = None, array_id: int | None = None):
        """
        Release a job.
        """
        squeue_cmd: str = 'squeue -r -t PD -u $USER -o '
        grep_cmd: str = f'grep "{job_id}_{array_id}$"' if array_id else f'grep "{job_id}$"'

        if dependency:
            subprocess.run(
                squeue_cmd +
                f'"scontrol update %i Dependency=afterok:{dependency}" | ' +
                grep_cmd + ' | sh',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        subprocess.run(
            squeue_cmd +
            '"scontrol release %i" | ' +
            grep_cmd + ' | sh',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
