import abc

from settings import Settings

class BaseStructurizer(abc.ABC):
    def __init__(self, settings: Settings, struct: dict = {}):
        self.settings = settings
        self.struct = struct
        
    @abc.abstractmethod
    async def extract(self, data):
        pass