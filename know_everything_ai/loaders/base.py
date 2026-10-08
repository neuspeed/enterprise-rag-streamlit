"""Base class for every source loader.

A loader's job is narrow: get bytes from a source into a local file, then hand
that file to a parser. Subclasses implement :meth:`transform`; downloading,
extension handling and scratch-file management live here so every format
behaves identically.

Local paths are copied into the scratch directory rather than parsed in place.
The parsers delete the file they are handed as cleanup, so passing a caller's
own file straight through would destroy the original.
"""

from __future__ import annotations

import abc
import asyncio
import shutil
import uuid
from pathlib import Path

from know_everything_ai.settings import Settings
from know_everything_ai.utils.llm_utils import LLMDriven
from know_everything_ai.utils.source_client import (
    SourceClient,
    SourceFetchError,
    guess_extension,
)


class BaseLoader(abc.ABC):
    #: Extensions this loader accepts. Empty means "anything".
    extensions: tuple[str, ...] = ()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.source_client = SourceClient(settings)

    @property
    def scratch_dir(self) -> Path:
        path = Path(self.settings.DATA_DIR) / "scratch"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _suffix_for(self, source: str, fallback_suffix: str = "") -> str:
        if not self.source_client.is_url(source):
            suffix = Path(str(source).replace("\\/", "/")).suffix
        else:
            suffix = guess_extension(source, default=fallback_suffix)
        return suffix or fallback_suffix

    def _filename_for(self, source: str, fallback_suffix: str = "") -> Path:
        suffix = self._suffix_for(source, fallback_suffix)
        if self.extensions and suffix.lower() not in self.extensions:
            expected = ", ".join(self.extensions)
            raise SourceFetchError(
                f"Unsupported file type {suffix!r} for {type(self).__name__}; "
                f"expected one of: {expected}"
            )
        return self.scratch_dir / f"{uuid.uuid4()}{suffix}"

    async def load(self, source: str) -> Path:
        """Materialise ``source`` as a local file and return its path.

        Subclasses that want the decoded text rather than a scratch file expose
        it as ``load_text``; overriding ``load`` to return something other than
        a ``Path`` would silently break every caller that expects a file.
        """
        if self.source_client.is_url(source):
            destination = self._filename_for(source)
            await self.source_client.fetch_to_file(source, destination)
            return destination

        if not self.settings.ALLOW_LOCAL_SOURCES:
            raise SourceFetchError(
                f"Local sources are disabled; {source!r} is not a URL"
            )

        path = Path(str(source).replace("\\/", "/")).expanduser()
        if not path.is_file():
            raise SourceFetchError(f"Source file not found: {path}")

        destination = self._filename_for(str(path))
        await asyncio.to_thread(shutil.copyfile, path, destination)
        return destination

    def validate(self, data: object) -> bool:
        """Whether ``data`` is usable by this loader."""
        return data is not None

    def llm_components(self) -> tuple[LLMDriven, ...]:
        """LLM-driven parsers this loader owns.

        The pipeline bills token spend across every component, and a VLM parse
        of a 200-page PDF costs more than the enrichment pass it feeds. Loaders
        that call a model must expose it here or it goes unbilled.
        """
        return ()

    @abc.abstractmethod
    async def transform(self, data_path: object):
        """Turn a downloaded source into ``RawElement`` objects."""
