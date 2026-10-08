"""Async resource for Flowise document stores.

All methods are coroutines and share a single connection pool owned by the
injected :class:`FlowiseTransport`.
"""

from __future__ import annotations

from typing import Any

from ..models import (
    ChunkResponse,
    DocumentChunk,
    DocumentStore,
    RefreshItem,
    RetrievalQuery,
    RetrievalResult,
    UpsertConfig,
    UpsertResult,
)
from ..transport import FlowiseTransport


class DocumentStoreResource:
    def __init__(self, transport: FlowiseTransport) -> None:
        self._transport = transport

    async def list_document_stores(self) -> list[DocumentStore]:
        data = await self._transport.request("GET", "/document-store/store")
        return [DocumentStore.from_dict(store) for store in data]

    async def get_document_store(self, store_id: str) -> DocumentStore:
        data = await self._transport.request(
            "GET", f"/document-store/store/{store_id}"
        )
        return DocumentStore.from_dict(data)

    async def find_document_store_by_name(self, name: str) -> DocumentStore | None:
        """Return the first store with this name, or None."""
        for store in await self.list_document_stores():
            if store.name == name:
                return store
        return None

    async def get_chunks(self, store_id: str, loader_id: str, page_no: int) -> ChunkResponse:
        data = await self._transport.request(
            "GET",
            f"/document-store/chunks/{store_id}/{loader_id}/{page_no}",
        )
        return ChunkResponse.from_dict(data)

    async def upsert_document(
        self,
        store_id: str,
        config: UpsertConfig,
        files: dict[str, Any] | None = None,
    ) -> UpsertResult:
        # Ingestion runs server-side and can take minutes on a large store, so
        # this call gets its own budget instead of the default request timeout.
        data = config.to_dict()
        if not files:
            # httpx only assembles a multipart body when files= is non-empty.
            # A config-only upsert therefore degrades silently to
            # x-www-form-urlencoded, which Flowise's multer-backed route does
            # not read. Filename-less parts keep it multipart.
            files = {key: (None, str(value)) for key, value in data.items()}
            data = {}

        result = await self._transport.request(
            "POST",
            f"/document-store/upsert/{store_id}",
            data=data,
            files=files,
            timeout=self._transport.upsert_timeout,
        )
        return UpsertResult.from_dict(result)

    async def refresh_document_store(
        self, store_id: str, items: list[RefreshItem]
    ) -> list[UpsertResult]:
        data = await self._transport.request(
            "POST",
            f"/document-store/refresh/{store_id}",
            json={"items": [item.to_dict() for item in items]},
        )
        if isinstance(data, list):
            return [UpsertResult.from_dict(item) for item in data]
        return [UpsertResult.from_dict(data)]

    async def query_vector_store(self, query: RetrievalQuery) -> RetrievalResult:
        data = await self._transport.request(
            "POST",
            "/document-store/vectorstore/query",
            json=query.to_dict(),
        )
        return RetrievalResult.from_dict(data)

    async def create_document_store(self, store: DocumentStore) -> DocumentStore:
        data = await self._transport.request(
            "POST",
            "/document-store/store",
            json=store.to_dict(),
        )
        return DocumentStore.from_dict(data)

    async def update_chunk(
        self, store_id: str, loader_id: str, chunk_id: str, chunk: DocumentChunk
    ) -> ChunkResponse:
        data = await self._transport.request(
            "PUT",
            f"/document-store/chunks/{store_id}/{loader_id}/{chunk_id}",
            json=chunk.to_dict(),
        )
        return ChunkResponse.from_dict(data)

    async def update_document_store(
        self, store_id: str, store: DocumentStore
    ) -> DocumentStore:
        data = await self._transport.request(
            "PUT",
            f"/document-store/store/{store_id}",
            json=store.to_dict(),
        )
        return DocumentStore.from_dict(data)

    async def delete_document_store(self, store_id: str) -> None:
        await self._transport.request(
            "DELETE",
            f"/document-store/store/{store_id}",
            expect_json=False,
        )

    async def delete_chunk(self, store_id: str, loader_id: str, chunk_id: str) -> None:
        await self._transport.request(
            "DELETE",
            f"/document-store/chunks/{store_id}/{loader_id}/{chunk_id}",
            expect_json=False,
        )

    async def delete_loader(self, store_id: str, loader_id: str) -> None:
        await self._transport.request(
            "DELETE",
            f"/document-store/loader/{store_id}/{loader_id}",
            expect_json=False,
        )

    async def delete_vector_store_data(self, store_id: str) -> None:
        await self._transport.request(
            "DELETE",
            f"/document-store/vectorstore/{store_id}",
            expect_json=False,
        )
