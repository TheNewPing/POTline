"""
Slurm dispatcher
"""

import time
from pathlib import Path
from enum import Enum

from simple_slurm import Slurm # type: ignore

from ..dispatcher import Dispatcher, SlurmCluster

class SlurmKW(Enum):
    """
    Supported cluster options.
    """
    MOD_CONDA = 'conda'
    ENV_PACE = 'env_pace'
    MOD_CUDA = 'cuda'

_cluster_paths: dict[SlurmCluster, Path] = {
    SlurmCluster.SNELLIUS: (Path(__file__).parent / 'template' / 'snellius').resolve()
}

_script_names: dict[SlurmKW, str] = {
    SlurmKW.MOD_CONDA: 'module_conda.sh',
    SlurmKW.MOD_CUDA: 'module_cuda.sh',
    SlurmKW.ENV_PACE: 'conda_pace.sh'
}

class SlurmDispatcher(Dispatcher):
    """
    Slurm command dispatcher.
    """
    def __init__(self, commands: list[str], options: dict | None = None):
        super().__init__(commands, options)
        self.job: Slurm = Slurm(**self.options)
        self._job_id: int = -1

    def dispatch(self):
        """
        Dispatch the command using Slurm.

        Returns:
            int: the job ID of the dispatched command.
        """
        for command in self.commands:
            self.job.add_cmd(command)
        self._job_id = self.job.sbatch()
        self.dispatched = True

    def wait(self):
        """
        Wait for the dispatched command to finish.
        """
        if self.dispatched:
            while True:
                self.job.squeue.update_squeue()
                if self._job_id not in self.job.squeue.jobs:
                    break
                time.sleep(10)
        else:
            raise ValueError("No command has been dispatched yet.")
