"""
Command dispatcher
"""

from abc import ABC, abstractmethod
from enum import Enum

class SupportedModel(Enum):
    """
    Supported models.
    """
    PACE = "pacemaker"
    MACE = "mace"
    GRACE = "grace"

class JobType(Enum):
    """
    Supported job types.
    """
    FIT = 'fit'
    INF = 'inf'
    DEEP = 'deep'
    SIM = 'sim'
    MAIN = 'main'

class SlurmCluster(Enum):
    """
    Supported clusters.
    """
    SNELLIUS = 'snellius'

class Dispatcher(ABC):
    """
    Base class for command dispatchers.
    """
    def __init__(self, commands: list[str],
                 options: dict | None = None,):
        self.commands = commands
        self.options = options
        self.dispatched = False

    @abstractmethod
    def dispatch(self):
        """
        Dispatch the command.
        """
    @abstractmethod
    def wait(self):
        """
        Wait for the dispatched command to finish.
        """
