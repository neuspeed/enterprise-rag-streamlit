"""Spreadsheet source loader (.csv, .xlsx, .xls)."""

from __future__ import annotations

from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.parsers.table_parser import TableParser


class TableDocumentLoader(BaseLoader):
    extensions = (".csv", ".xlsx", ".xls")

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._parser = TableParser()

    async def transform(self, data_path):
        local_path = await self.load(str(data_path))
        return await self._parser.parse(local_path)
