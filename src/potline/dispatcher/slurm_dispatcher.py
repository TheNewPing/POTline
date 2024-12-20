"""
Slurm dispatcher
"""

import time
import subprocess
import csv
from io import StringIO
from typing import Any

from simple_slurm import Slurm # type: ignore

class SlurmDispatcher():
    """
    Slurm command dispatcher.
    """
    jobs: dict[int, Any] = {}

    def __init__(self, commands: list[str], options: dict | None = None):
        self.commands = commands
        self.options = options
        self.dispatched = False
        self.job: Slurm = Slurm(**self.options)
        self._job_id: int = -1

    def dispatch(self) -> int:
        """
        Dispatch the command using Slurm.
        """
        for command in self.commands:
            self.job.add_cmd(command)
        self._job_id = self.job.sbatch()
        self.dispatched = True
        return self._job_id

    def wait(self):
        """
        Wait for the dispatched command to finish.
        """
        if self.dispatched:
            while True:
                SlurmDispatcher._update_squeue()
                if self._job_id not in SlurmDispatcher.jobs:
                    break
                time.sleep(10)
        else:
            raise ValueError("No command has been dispatched yet.")

    @staticmethod
    def _update_squeue():
        '''Refresh the information from the current queue for the current user'''
        result = subprocess.run(["squeue", "--me", "-o",
                                 '%.18F, %.9P, %.8j, %.8u, %.2t, %.10M, %.6D, %R'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        if result.returncode != 0:
            raise RuntimeError(f"Error running squeue: {result.stderr.strip()}")

        print(result.stdout.strip())
        SlurmDispatcher.jobs = SlurmDispatcher._parse_output(result.stdout.strip())

    @staticmethod
    def _parse_output(output):
        '''
        Converts the stdout into a python dictionary
        each key is a jobid as integer
        '''
        csv_file = StringIO(output)
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='"', skipinitialspace=True)
        jobs = {}
        for row in reader:
            print(row)
            jobs[int(row["ARRAY_JOB_ID"])] = row
        return jobs
