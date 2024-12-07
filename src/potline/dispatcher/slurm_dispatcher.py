"""
Slurm dispatcher
"""

import time
from pathlib import Path
from enum import Enum

from simple_slurm import Slurm

from .dispatcher import Dispatcher

SNELLIUS_KW: str = 'snellius'

MOD_CONDA_KW: str = 'conda'
ACTIVATE_ENV_KW: str = 'env'
MOD_CUDA_KW: str = 'cuda'

class SlurmCluster(Enum):
    """
    Supported clusters.
    """
    SNELLIUS = 'snellius'

class SlurmKW(Enum):
    """
    Supported cluster options.
    """
    MOD_CONDA = 'conda'
    ENV_PACE = 'env_pace'
    MOD_CUDA = 'cuda'

_cluster_paths: dict[str, Path] = {
    SlurmCluster.SNELLIUS: (Path(__file__).parent / 'template' / 'snellius').resolve()
}

_script_names: dict[str, str] = {
    SlurmKW.MOD_CONDA: 'module_conda.sh',
    SlurmKW.MOD_CUDA: 'module_cuda.sh',
    SlurmKW.ENV_PACE: 'conda_pace.sh'
}

class SlurmDispatcher(Dispatcher):
    """
    Slurm command dispatcher.
    """
    def __init__(self, command: str, options: dict | None = None):
        super().__init__(command, options)
        self._prepare_cluster()
        self.job: Slurm = Slurm(self.options)
        self._job_id: int = -1

    def dispatch(self):
        """
        Dispatch the command using Slurm.

        Returns:
            int: the job ID of the dispatched command.
        """
        self.dispatched = True
        self._job_id = self.job.sbatch(self.command)

    def wait(self):
        """
        Wait for the dispatched command to finish.
        """
        if self.dispatched:
            while True:
                self.job.squeue.update_squeue()
                if self._job_id not in self.job.squeue:
                    break
                time.sleep(10)
        else:
            raise ValueError("No command has been dispatched yet.")

    def _prepare_cluster(self):
        """
        Prepare the job options for the cluster.
        """
        if self.options:
            for key in self.options.keys():
                # identify the cluster
                if key in SlurmCluster.__members__:
                    cluster_path: Path = _cluster_paths[SlurmCluster[key]]
                    cluster_options: dict = self.options[key]
                    for option in cluster_options.keys():
                        # add cluster options
                        if option in SlurmKW.__members__ and cluster_options[option]:
                            script_path: Path = cluster_path / _script_names[SlurmKW[option]]
                            self.job.add_cmd(f"source {script_path}")
                    # remove the cluster options from the options dict
                    del self.options[key]
