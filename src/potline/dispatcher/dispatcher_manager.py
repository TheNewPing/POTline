"""
dispatcher factory
"""

from pathlib import Path

from .slurm_preset import get_slurm_options, JobType
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
                dependency: int | None = None):
        """
        Create a dispatcher based on the options.

        Args:
            - commands: commands to run
            - out_path: path to the output directory
            - job_config: job configuration
            - array_ids: array ids to run
            - dependency: job dependency

        Returns:
            Dispatcher: the dispatcher to use.
        """
        slurm_dict = job_config.slurm_watcher if self._job_type == JobType.WATCH.value \
            else job_config.slurm_opts

        options = get_slurm_options(
            self._cluster, self._job_type, out_path, self._model,
            slurm_dict, array_ids, dependency)
        source_cmds = [f'source {cmd}' for cmd in job_config.modules]
        py_cmds = [f'python {cmd}' for cmd in job_config.py_scripts] if \
            self._job_type == JobType.WATCH.value else []
        tot_cmds = source_cmds + py_cmds + commands
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
