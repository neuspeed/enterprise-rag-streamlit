# Base loader class for all loaders. This class provides the basic structure and methods that all loaders should implement.

import abc

import uuid
from pathlib import Path

from utils.owncloud_client import OwnCloudClient
from settings import Settings

class BaseLoader(abc.ABC):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.owncloud_client = OwnCloudClient(
            base_url=settings.OWNCLOUD_BASE_URL,
            username=settings.OWNCLOUD_USER,
            password=settings.OWNCLOUD_PASSWORD
        )
    
    async def load(self, source) -> Path:
        """
        Load data from the given source.

        Args:
            source: The source from which to load data. This could be a file path, URL, or any other identifier.

        Returns:
            The loaded data path.
            
        """
        # Загрузка файла из OwnCloud во временное место
        local_path: Path = Path(self.settings.LOCAL_PATH)
        local_path.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path / f"{uuid.uuid4()}{Path(source).suffix}"
        try:
            await self.owncloud_client.download_to_file(
                remote_url=source, dest_path=tmp_path
            )
        except Exception as e:
            raise(e)
        return tmp_path  # Возвращаем путь к загруженному файлу для дальнейшей обработки

    @abc.abstractmethod
    def validate(self, data):
        """
        Validate the loaded data.

        Args:
            data: The data to validate.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        pass

    @abc.abstractmethod
    def transform(self, data_path):
        """
        Transform the loaded data into the desired format.

        Args:
            data_path: The path to the data to transform.

        Returns:
            The transformed data.
        """
        pass