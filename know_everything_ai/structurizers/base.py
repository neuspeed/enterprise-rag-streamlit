import abc

from know_everything_ai.settings import Settings


class BaseStructurizer(abc.ABC):
    """Extracts structured records from already-classified text.

    Subclasses take only settings. The class previously accepted a ``struct``
    argument with a mutable default and stored it unused, so every instance
    shared one dict and it never carried data anyway.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abc.abstractmethod
    async def extract(self, data):
        """Return the records found in ``data``."""
        raise NotImplementedError
