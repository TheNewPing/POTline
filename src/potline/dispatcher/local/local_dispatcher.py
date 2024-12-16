"""
Local dispatcher
"""

import subprocess

from ..dispatcher import Dispatcher

class LocalDispatcher(Dispatcher):
    """
    Local command dispatcher.
    """
    def dispatch(self) -> int:
        """
        Dispatch the command locally.

        Returns:
            int: the job ID of the dispatched command. (0 if no job is dispatched)
        """
        subprocess.run(self.commands[-1], check=True)
        self.dispatched = True
        return 0

    def wait(self):
        """
        Wait for the dispatched command to finish.
        """
        if not self.dispatched:
            raise RuntimeError("No job dispatched to wait for.")
