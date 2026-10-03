# Word Document Loader

from loaders.base import BaseLoader

from parsers.word_parser import WordParser


class WordDocumentLoader(BaseLoader): 
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
        return await WordParser().parse(local_path)