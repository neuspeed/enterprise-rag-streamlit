"""Source type to loader mapping.

Loaders are created per pipeline instance rather than at import time. The old
module-level ``Settings()`` plus a dict of pre-built singletons meant
configuration was read once, at import, and the same loader objects were shared
across every concurrent job.
"""

from __future__ import annotations

from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.loaders.html_loader import HTMLDocumentLoader
from know_everything_ai.loaders.pdf_loader import PDFDocumentLoader
from know_everything_ai.loaders.table_loader import TableDocumentLoader
from know_everything_ai.loaders.text_loader import TextDocumentLoader
from know_everything_ai.loaders.word_loader import WordDocumentLoader
from know_everything_ai.settings import Settings
from know_everything_ai.utils.source_client import SourceFetchError

LOADER_TYPES: dict[str, type[BaseLoader]] = {
    "docx": WordDocumentLoader,
    "pdf": PDFDocumentLoader,
    "csv": TableDocumentLoader,
    "xlsx": TableDocumentLoader,
    "xls": TableDocumentLoader,
    "html": HTMLDocumentLoader,
    "htm": HTMLDocumentLoader,
    "txt": TextDocumentLoader,
    "text": TextDocumentLoader,
    "md": TextDocumentLoader,
    "markdown": TextDocumentLoader,
}

SUPPORTED_SOURCE_TYPES = frozenset(LOADER_TYPES)


def create_loader(source_type: str, settings: Settings) -> BaseLoader:
    """Build the loader for a source type such as ``"pdf"`` or ``"xlsx"``."""
    try:
        loader_class = LOADER_TYPES[source_type.lower().lstrip(".")]
    except KeyError:
        supported = ", ".join(sorted(SUPPORTED_SOURCE_TYPES))
        raise SourceFetchError(
            f"Unsupported source type {source_type!r}; supported: {supported}"
        ) from None
    return loader_class(settings)
