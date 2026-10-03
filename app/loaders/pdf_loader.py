# PDF Document Loader

from loaders.base import BaseLoader

from parsers.pdf_parser import PDFParser


class PDFDocumentLoader(BaseLoader):    
    def validate(self):
        pass
             
    async def transform(self, data_path):
        """
        Transform the loaded PDF document data into the desired format.

        Args:
            data_path: The path to document for transform.

        Returns:
            The transformed data.
        """
        local_path = await self.load(data_path)
        return await PDFParser(
            api_url=self.settings.VLM_API_URL,
            api_key=self.settings.VLM_API_KEY,
            model=self.settings.VLM_MODEL_NAME,
            system_prompt=self.settings.VLM_PROMPT_TEMPLATE
            ).parse(local_path)