"""``append_chunks``: the update half of the built-in store's write API.

Replacing a knowledge base and extending one are different promises, and the
in-memory store is where the difference is cheapest to pin down: an append
must leave every previously stored fragment in place.
"""

from __future__ import annotations

import pytest

from know_everything_ai.stores.base import Chunk
from know_everything_ai.stores.embeddings import FakeEmbedder
from know_everything_ai.stores.memory import MemoryVectorStore


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore(FakeEmbedder(dim=8))


@pytest.mark.asyncio
async def test_append_keeps_the_previous_fragments(store: MemoryVectorStore) -> None:
    await store.replace_chunks(1, [Chunk(content="старый")])

    added = await store.append_chunks(1, [Chunk(content="новый")])

    assert added == 1
    assert await store.count_chunks(1) == 2
    contents = [hit.content for hit in await store.list_chunks(1)]
    assert contents == ["старый", "новый"]


@pytest.mark.asyncio
async def test_append_to_an_unknown_kb_creates_it(store: MemoryVectorStore) -> None:
    """The registry always resolves the id first, but an append into an empty
    row set must not silently drop the fragments either."""
    added = await store.append_chunks(7, [Chunk(content="x")])

    assert added == 1
    assert await store.count_chunks(7) == 1


@pytest.mark.asyncio
async def test_append_of_nothing_is_a_noop(store: MemoryVectorStore) -> None:
    await store.replace_chunks(1, [Chunk(content="старый")])

    added = await store.append_chunks(1, [])

    assert added == 0
    assert await store.count_chunks(1) == 1


@pytest.mark.asyncio
async def test_replace_still_overwrites_what_append_left(
    store: MemoryVectorStore,
) -> None:
    """The two modes coexist: a fresh run keeps its old contract."""
    await store.replace_chunks(1, [Chunk(content="старый")])
    await store.append_chunks(1, [Chunk(content="добавленный")])

    await store.replace_chunks(1, [Chunk(content="свежая версия")])

    contents = [hit.content for hit in await store.list_chunks(1)]
    assert contents == ["свежая версия"]
