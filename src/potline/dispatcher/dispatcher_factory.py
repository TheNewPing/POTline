"""
dispatcher factory
"""

from pathlib import Path

from .dispatcher import Dispatcher
from .local import LocalDispatcher
from .slurm import SlurmDispatcher, get_slurm_commands, get_slurm_options

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
                 cluster: str | None = None):
        self._job_type = job_type
        self._model = model
        self._cluster = cluster
        self._dispatcher: Dispatcher | None = None

    def set_job(self, commands: list[str], out_path: Path,
                array_ids: list[int] | None = None,
                n_cpu: int | None = None,
                email: str | None = None):
        """
        Create a dispatcher based on the options.

        Args:
            - commands: commands to run
            - out_path: path to the output directory
            - array_ids: array ids to run
            - n_cpu: number of CPUs to use
            - email: email to send the job results to

        Returns:
            Dispatcher: the dispatcher to use.
        """
        options: dict | None = None
        tot_cmds: list[str] = commands

        if self._cluster:
            options = get_slurm_options(
                self._cluster, self._job_type, out_path, self._model, n_cpu, email, array_ids)
            tot_cmds = get_slurm_commands(self._cluster, self._job_type, self._model) + commands
            self._dispatcher = SlurmDispatcher(tot_cmds, options)

        self._dispatcher = LocalDispatcher(tot_cmds, options)

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
