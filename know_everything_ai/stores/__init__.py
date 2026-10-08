"""Built-in persistence for the read side: embeddings, vectors, registry.

Flowise stays an optional *adapter* alongside these rather than a provider of
them. Everything here works with ``FLOWISE_ENABLED=false``, which is what the
standing development stack runs, and the flowise document-store id is recorded
as an attribute of a knowledge base instead of being the only way to reach one.
"""

from __future__ import annotations

from know_everything_ai.settings import Settings
from know_everything_ai.stores.api_keys import ApiKeyStore, IssuedKey
from know_everything_ai.stores.base import Chunk, Embedder, SearchHit, VectorStore
from know_everything_ai.stores.canvas import (
    canvas_sections,
    canvas_to_chunks,
    split_oversized,
    strip_fence,
)
from know_everything_ai.stores.embeddings import FakeEmbedder, FastembedEmbedder
from know_everything_ai.stores.memory import MemoryStoreContext
from know_everything_ai.stores.pgvector import PgVectorStore
from know_everything_ai.stores.registry import KnowledgeBaseRegistry, SuccessRecord

__all__ = [
    "ApiKeyStore",
    "Chunk",
    "Embedder",
    "FakeEmbedder",
    "FastembedEmbedder",
    "IssuedKey",
    "MemoryStoreContext",
    "KnowledgeBaseRegistry",
    "PgVectorStore",
    "SearchHit",
    "StoreContext",
    "SuccessRecord",
    "VectorStore",
    "canvas_sections",
    "canvas_to_chunks",
    "split_oversized",
    "strip_fence",
]


class StoreContext:
    """Everything the pipeline needs to persist a knowledge base.

    Constructed without touching the network: the embedder downloads its weights
    and the pool opens on first use, so building a pipeline in a test, or in a
    process that only previews, does not require either a database or 220 MB of
    model files.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = FastembedEmbedder(settings)
        self.vector_store = PgVectorStore(settings, self.embedder)
        self.registry = KnowledgeBaseRegistry(settings)

    async def aclose(self) -> None:
        # The store owns the embedder and closes it; closing it here too would
        # double-release the ONNX session.
        await self.vector_store.aclose()
        await self.registry.aclose()
