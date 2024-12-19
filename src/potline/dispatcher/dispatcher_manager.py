"""
dispatcher factory
"""

from pathlib import Path

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
                array_ids: list[int] | None = None):
        """
        Create a dispatcher based on the options.

        Args:
            - commands: commands to run
            - out_path: path to the output directory
            - job_config: job configuration
            - array_ids: array ids to run

        Returns:
            Dispatcher: the dispatcher to use.
        """
        options = get_slurm_options(
            self._cluster, self._job_type, out_path, self._model, job_config.slurm_opts, array_ids)
        source_cmds = [f'source {cmd}' for cmd in job_config.modules]
        py_cmds = [f'python {cmd}' for cmd in job_config.py_scripts]
        tot_cmds = source_cmds + py_cmds + commands
        self._dispatcher = SlurmDispatcher(tot_cmds, options)

    def dispatch_job(self):
        """
        Dispatch the job.
        """
        self._dispatcher.dispatch()

    def wait_job(self):
        """
        Wait for the job to finish.
        """
        self._dispatcher.wait()
