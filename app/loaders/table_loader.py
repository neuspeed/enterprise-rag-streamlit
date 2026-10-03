# Table Document Loader

from loaders.base import BaseLoader

from parsers.table_parser import TableParser

class TableDocumentLoader(BaseLoader):    
    def validate(self, data):
        return super().validate(data)
             
    async def transform(self, data_path):
        """
        Transform the loaded Word document data into the desired format.

        Args:
            data_path: The path to document for transform.

        Returns:
            The transformed data.
        """
        local_path = await self.load(data_path)
        return await TableParser().parse(local_path)