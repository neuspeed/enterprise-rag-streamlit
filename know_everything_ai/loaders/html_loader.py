"""HTML source loader.

Pages arrive either as a live URL or as an already-downloaded file. The fetch
runs through :class:`SourceClient`, so HTML gets the same SSRF guard, redirect
re-validation, WebDAV credentials and size limit as every other format. The
earlier version issued its own unguarded ``aiohttp.get`` and followed redirects
blindly, which made an arbitrary URL fetch a way to reach anything the container
could see.

Failures are raised, not swallowed. Returning ``None`` on error made a broken
page indistinguishable from an empty one, so jobs silently produced knowledge
bases with holes in them.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import structlog

from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.parsers.html_parser import HTMLParser
from know_everything_ai.utils.decoding import decode_bytes
from know_everything_ai.utils.source_client import SourceFetchError, guess_extension

log = structlog.get_logger("html_loader")


class HTMLDocumentLoader(BaseLoader):
    extensions = (".html", ".htm", ".xhtml")

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._parser = HTMLParser()

    def validate(self, data: object) -> bool:
        if isinstance(data, (str, bytes)):
            return bool(data)
        if isinstance(data, Path):
            return data.exists()
        return False

    def _filename_for(self, source: str, fallback_suffix: str = ".html") -> Path:
        # Web pages rarely end in a real extension (think ``/pricing`` or
        # ``?id=4``), so anything unrecognised is fetched as HTML rather than
        # rejected outright.
        suffix = guess_extension(source)
        if suffix.lower() not in self.extensions:
            suffix = fallback_suffix
        return self.scratch_dir / f"{uuid.uuid4()}{suffix}"

    async def load_text(self, source: str) -> str:
        """Return the raw HTML for ``source`` as text."""
        if not self.source_client.is_url(source):
            path = Path(str(source).replace("\\/", "/"))
            if not path.exists():
                raise SourceFetchError(f"HTML file not found: {path}")
            payload = await asyncio.to_thread(path.read_bytes)
            return decode_bytes(payload)

        destination = self._filename_for(source)
        try:
            result = await self.source_client.fetch_to_file(source, destination)
            payload = await asyncio.to_thread(destination.read_bytes)
        finally:
            await asyncio.to_thread(destination.unlink, missing_ok=True)

        log.debug("html_fetched", source=source, size=result.size)
        return decode_bytes(payload, result.content_type)

    async def transform(self, data_path: object):
        """Parse HTML into ``RawElement`` objects."""
        source = str(data_path)
        html = await self.load_text(source)
        return await self._parser.parse(html, source=source)
