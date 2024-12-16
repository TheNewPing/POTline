"""
Slurm dispatcher
"""

import time

from simple_slurm import Slurm # type: ignore

from ..dispatcher import Dispatcher

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
