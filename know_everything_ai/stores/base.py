"""Storage interfaces for the read side of a knowledge base.

Kept behind protocols for two reasons that are not aesthetic: the pipeline must
be testable without a running Postgres, and the embedding provider has to be
swappable between the local ONNX model and an OpenAI-compatible endpoint without
the query path noticing which one answered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Chunk:
    """One retrievable fragment, before it is embedded."""

    content: str
    category: str | None = None
    source: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """A ranked fragment with enough context to be cited."""

    chunk_id: int
    content: str
    score: float
    category: str | None = None
    source: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Query and document text may need different
    treatment (the ``query:``/``passage:`` convention), which is why they are
    separate methods rather than one call with a prefix argument."""

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """The built-in store. Flowise is an optional adapter with the same shape
    from the pipeline's point of view, not this interface's implementation."""

    async def replace_chunks(self, kb_id: int, chunks: Sequence[Chunk]) -> int:
        """Swap in the full fragment set for a knowledge base.

        Replace rather than upsert: a re-run must not leave fragments from the
        version of the document it superseded, and doing it in one transaction
        means a failure half-way leaves the previous set intact rather than a
        knowledge base missing an unseen portion of itself.
        """
        ...

    async def append_chunks(self, kb_id: int, chunks: Sequence[Chunk]) -> int:
        """Add fragments to a knowledge base without touching the existing ones.

        The update half of :meth:`replace_chunks`: an operator extending a
        knowledge base with new files must not lose the fragments already
        there, and re-embedding them to rewrite the whole set would charge for
        work that did not change.
        """
        ...

    async def search(
        self, kb_id: int, query: str, limit: int
    ) -> list[SearchHit]: ...

    async def count_chunks(self, kb_id: int) -> int: ...

    async def aclose(self) -> None: ...
