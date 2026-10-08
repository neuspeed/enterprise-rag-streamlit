"""Wire-level checks for the Flowise HTTP layer.

These assert on what actually goes on the wire: the upsert endpoint is a
multipart form, and getting its Content-Type wrong is invisible in unit tests
that mock httpx but fatal against a real Flowise.
"""

import httpx
import pytest

from know_everything_ai.flowise.models.document_store import UpsertConfig
from know_everything_ai.flowise.resources.document_store import DocumentStoreResource
from know_everything_ai.flowise.transport import FlowiseTransport

_EMPTY = {"updatedDocuments": [], "insertedDocuments": []}
_LIST: list[dict] = []


def _transport(handler) -> tuple[FlowiseTransport, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://flowise",
    )
    return FlowiseTransport("http://flowise", "secret", client=client), client


@pytest.mark.asyncio
async def test_upsert_sends_multipart_with_boundary() -> None:
    """The upsert endpoint must not be labelled application/json.

    httpx emits the multipart boundary itself; an explicit Content-Type
    suppresses that header and Flowise's multipart parser cannot find a
    boundary at all.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_EMPTY)

    transport, client = _transport(handler)
    api = DocumentStoreResource(transport)

    await api.upsert_document(
        store_id="store-1",
        config=UpsertConfig(loader={"text-loader": {}}),
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    await client.aclose()

    assert len(seen) == 1
    content_type = seen[0].headers["content-type"]
    assert content_type.startswith("multipart/form-data"), content_type
    assert "boundary=" in content_type, content_type

    body = seen[0].content
    assert b'name="loader"' in body
    assert b'name="file"' in body


@pytest.mark.asyncio
async def test_upsert_without_files_still_multipart() -> None:
    """Flowise expects multipart even when no file part is attached."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_EMPTY)

    transport, client = _transport(handler)
    api = DocumentStoreResource(transport)

    await api.upsert_document(
        store_id="store-1",
        config=UpsertConfig(loader={"text-loader": {}}),
    )
    await client.aclose()

    assert seen[0].headers["content-type"].startswith("multipart/form-data")


@pytest.mark.asyncio
async def test_json_endpoints_keep_json_content_type() -> None:
    """Everything that is not an upsert stays plain JSON."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_LIST)

    transport, client = _transport(handler)
    api = DocumentStoreResource(transport)

    await api.list_document_stores()
    await client.aclose()

    assert seen[0].headers["content-type"] == "application/json"
    assert seen[0].headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_upsert_serialises_nested_config_as_json_strings() -> None:
    """Flowise parses the sub-config objects out of their string fields."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_EMPTY)

    transport, client = _transport(handler)
    api = DocumentStoreResource(transport)

    config = UpsertConfig(
        loader={"text-loader": {}},
        embedding={"model": "text-embedding-3-large"},
        vectorStore={"type": "pinecone"},
    )
    await api.upsert_document(store_id="store-1", config=config)
    await client.aclose()

    body = seen[0].content
    # Nested sections are JSON-encoded in the part body. A raw dict would have
    # produced an unquoted [object Object] instead.
    assert b'name="loader"' in body
    assert b'{"text-loader": {}}' in body
    assert b'{"model": "text-embedding-3-large"}' in body
    assert b"[object Object]" not in body
