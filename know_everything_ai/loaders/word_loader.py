"""Word document source loader (.docx)."""

from __future__ import annotations

from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.parsers.word_parser import WordParser


class WordDocumentLoader(BaseLoader):
    extensions = (".docx",)

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._parser = WordParser()

    async def transform(self, data_path):
        local_path = await self.load(str(data_path))
        return await self._parser.parse(local_path)
