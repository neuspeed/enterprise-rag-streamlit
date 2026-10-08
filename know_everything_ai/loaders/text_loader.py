"""Plain-text source loader.

``.txt`` was listed as a supported format but had no loader, so a knowledge base
of plain-text files failed with "Unsupported source type". Text has no structure
to extract, so the bytes are decoded and split into blocks by the same Markdown
converter the HTML path uses — which means a ``.txt`` containing Markdown, or
one that is just prose, both come out as properly typed fragments.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.schemas import DocCategory, RawElement
from know_everything_ai.utils.decoding import decode_bytes
from know_everything_ai.utils.markdown_to_raw_elements import MarkdownToRawElements
from know_everything_ai.utils.source_client import SourceFetchError

log = structlog.get_logger("text_loader")


class TextDocumentLoader(BaseLoader):
    extensions = (".txt", ".text", ".md", ".markdown")

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._splitter = MarkdownToRawElements()

    async def load_text(self, source: str) -> str:
        """Return the decoded text of ``source``."""
        if not self.source_client.is_url(source):
            path = Path(str(source).replace("\\/", "/")).expanduser()
            if not path.is_file():
                raise SourceFetchError(f"Text file not found: {path}")
            return decode_bytes(await asyncio.to_thread(path.read_bytes))

        destination = self._filename_for(source)
        try:
            result = await self.source_client.fetch_to_file(source, destination)
            payload = await asyncio.to_thread(destination.read_bytes)
        finally:
            await asyncio.to_thread(destination.unlink, missing_ok=True)

        log.debug("text_fetched", source=source, size=result.size)
        return decode_bytes(payload, result.content_type)

    async def transform(self, data_path: object) -> list[RawElement]:
        source = str(data_path)
        text = await self.load_text(source)
        if not text.strip():
            return [
                RawElement(
                    content=f"Файл {source} не содержит текста",
                    source=source,
                    category=DocCategory.UNKNOWN,
                )
            ]
        # The splitter walks the whole file with a dozen regexes; keep it off
        # the event loop like every other CPU-bound parse.
        return await asyncio.to_thread(
            self._splitter.convert, text, source=source
        )
