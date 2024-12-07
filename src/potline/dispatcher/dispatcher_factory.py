"""
dispatcher factory
"""

from .dispatcher import Dispatcher
from .local_dispatcher import LocalDispatcher
from .slurm_dispatcher import SlurmDispatcher

def create_dispatcher(command: str, hpc: bool, options: dict | None = None) -> Dispatcher:
    """
    Create a dispatcher based on the options.

    Args:
        - command: the command to dispatch.
        - hpc: whether to use HPC.
        - options: dispatch settings.

    Returns:
        Dispatcher: the dispatcher to use.
    """
    if hpc:
        return SlurmDispatcher(command, options)
    return LocalDispatcher(command, options)
