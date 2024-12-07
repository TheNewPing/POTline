"""
Local dispatcher
"""

import subprocess

from .dispatcher import Dispatcher

class LocalDispatcher(Dispatcher):
    """
    Local command dispatcher.
    """
    def dispatch(self) -> int:
        """
        Dispatch the command locally.

        Args:
            - command: the command to dispatch.
            - options: dispatch settings. (ignored)

        Returns:
            int: the job ID of the dispatched command. (0 if no job is dispatched)
        """
        subprocess.run(self.command, check=True)
        return 0

    def wait(self):
        """
        Wait for the dispatched command to finish.
        """
