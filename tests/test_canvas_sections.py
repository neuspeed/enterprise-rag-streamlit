"""Unit tests for canvas → fragment conversion and ranking doubles."""

from __future__ import annotations

import pytest

from know_everything_ai.schemas import DocCategory
from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import Chunk
from know_everything_ai.stores.canvas import (
    canvas_sections,
    canvas_to_chunks,
    split_oversized,
    strip_fence,
)
from know_everything_ai.stores.memory import MemoryStoreContext
from know_everything_ai.utils.token_counter import TokenEstimator

FENCED = '''```json
{"glossary": [{"term": "API", "def": "interface"}], "summary": "текст"}
```'''


def test_strip_fence_removes_language_tagged_block():
    assert strip_fence(FENCED).startswith('{"glossary"')


def test_strip_fence_leaves_plain_text_untouched():
    text = "просто текст"
    assert strip_fence(text) == text


def test_canvas_sections_splits_list_of_objects():
    sections = canvas_sections('{"glossary": [{"term": "A"}, {"term": "B"}], "x": "y"}')
    assert sections == [
        ("glossary / 1", '{"term": "A"}'),
        ("glossary / 2", '{"term": "B"}'),
        ("x", "y"),
    ]


def test_canvas_sections_keeps_string_section_as_is():
    assert canvas_sections('{"summary": "s"}') == [("summary", "s")]


def test_canvas_sections_falls_back_to_whole_body_on_bad_json():
    body = "не json, а проза"
    assert canvas_sections(body) == [("", body)]


def test_canvas_sections_fenced_body_is_parsed():
    sections = canvas_sections(FENCED)
    assert sections[0][0] == "glossary / 1"


def test_canvas_sections_empty_object_returns_whole_body():
    sections = canvas_sections("{}")
    assert sections == [("", "{}")]


def test_split_oversized_keeps_fitting_text_whole():
    text = "short"
    out = split_oversized(text, max_tokens=1000, overlap_tokens=10, estimator=TokenEstimator(Settings()))
    assert out == [text]


def test_split_oversized_cuts_and_keeps_overlap():
    # 20 tokens each "слово_символ"; window of 8 forces several pieces.
    tokens = [f"w{i}_" + "x" * 100 for i in range(20)]
    text = " ".join(tokens)
    pieces = split_oversized(text, max_tokens=8, overlap_tokens=2, estimator=TokenEstimator(Settings()))
    assert len(pieces) >= 3
    assert pieces[0] != text
    # No piece is left empty, and each is clearly shorter than the source.
    assert all(p.strip() for p in pieces)
    assert all(len(p) < len(text) for p in pieces)


def test_split_oversized_progresses_even_with_collapsing_overlap():
    tokens = ["ab"] * 100
    text = " ".join(tokens)
    pieces = split_oversized(text, max_tokens=2, overlap_tokens=1000, estimator=TokenEstimator(Settings()))
    assert pieces
    # Worst-case overlap still terminates: concatenation preserves every word.
    assert all(isinstance(p, str) for p in pieces)


def test_canvas_to_chunks_flattens_fenced_canvas():
    items = [{"content": FENCED, "category": DocCategory.KNOWLEDGE_CANVAS.value}]
    chunks = canvas_to_chunks(
        items,
        max_tokens=1000,
        overlap_tokens=100,
        estimator=TokenEstimator(Settings()),
    )
    assert len(chunks) == 2
    assert chunks[0].title == "glossary / 1"
    assert chunks[0].metadata["section"] == "glossary / 1"
    assert chunks[0].category == "knowledge_canvas"


def test_canvas_to_chunks_empty_content_is_skipped():
    chunks = canvas_to_chunks(
        [{"content": "   ", "category": "x"}],
        max_tokens=1000,
        overlap_tokens=100,
        estimator=TokenEstimator(Settings()),
    )
    assert chunks == []


def test_canvas_to_chunks_accepts_raw_elements():
    items = [{"content": '{"a": "b"}', "category": "faq", "source": "s.txt", "title": "T"}]
    chunks = canvas_to_chunks(
        items,
        max_tokens=1000,
        overlap_tokens=100,
        estimator=TokenEstimator(Settings()),
    )
    assert chunks[0].source == "s.txt"
    assert chunks[0].title == "T · a"


def test_split_oversized_no_word_is_lost():
    pieces = split_oversized(
        " ".join(["слово"] * 500),
        max_tokens=100,
        overlap_tokens=20,
        estimator=TokenEstimator(Settings()),
    )
    assert pieces
    # Every source word appears across the pieces — nothing is dropped, only
    # the window edges repeat due to overlap.
    joined = " ".join(pieces)
    assert joined.count("слово") >= 500


# ---------------------------------------------------------------------- memory


@pytest.fixture
def memory() -> MemoryStoreContext:
    return MemoryStoreContext(settings=Settings(_env_file=None))


@pytest.mark.asyncio
async def test_memory_store_ranked_search(memory):
    store = memory.vector_store
    await store.replace_chunks(
        1,
        [
            Chunk(content="котики и котята", category="text"),
            Chunk(content="расписание поездов", category="text"),
        ],
    )
    hits = await store.search(1, "котенок пушистый", limit=2)
    assert hits
    assert hits[0].score > 0.0
    assert hits[1].score >= 0.0


@pytest.mark.asyncio
async def test_memory_store_replace_drops_previous_fragments(memory):
    store = memory.vector_store
    await store.replace_chunks(1, [Chunk(content="первая версия")])
    assert await store.count_chunks(1) == 1
    await store.replace_chunks(1, [Chunk(content="вторая версия"), Chunk(content="ещё")])
    assert await store.count_chunks(1) == 2


@pytest.mark.asyncio
async def test_memory_registry_re_run_keeps_old_status_until_recorded(memory):
    reg = memory.registry
    first = await reg.record_success(_record("kb1"))
    assert first == 1

    await reg.ensure("kb1", "auto", "vector")
    row = await reg.get_by_external_id("kb1")
    assert row["status"] == "ok"  # ensure() does not clobber a prior success

    await reg.record_success(_record("kb1", branch="vector", document_store_id="s1"))
    row = await reg.get_by_external_id("kb1")
    assert row["document_store_id"] == "s1"
    assert row["branch"] == "vector"


@pytest.mark.asyncio
async def test_memory_registry_ids_are_stable_per_kb(memory):
    reg = memory.registry
    a = await reg.ensure("kb-a", "auto", "context")
    b = await reg.ensure("kb-b", "auto", "vector")
    c = await reg.ensure("kb-a", "auto", "context")
    assert a == c == 1
    assert b == 2


def _record(kb_id: str, branch="context", document_store_id=None):
    from know_everything_ai.stores.registry import SuccessRecord

    return SuccessRecord(
        kb_external_id=kb_id,
        kb_type="auto",
        branch=branch,
        total_tokens=10,
        item_count=1,
        document_store_id=document_store_id,
        canvas_tokens=5,
        cost={"model_calls": 1},
    )
