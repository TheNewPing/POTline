"""
Command dispatcher
"""

from abc import ABC, abstractmethod

class Dispatcher(ABC):
    """
    Base class for command dispatchers.

    Args:
        - command: the command to dispatch.
        - options: dispatch settings.
    """
    def __init__(self, command: str, options: dict | None = None):
        self.command = command
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
