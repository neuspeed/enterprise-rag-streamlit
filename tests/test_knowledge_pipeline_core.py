import json
from unittest.mock import AsyncMock, Mock

import pytest

from know_everything_ai.pipeline import KnowledgePipeline, QuotaExceeded
from know_everything_ai.schemas import Classified, DocCategory, Payload, RawElement
from know_everything_ai.settings import Settings


# Helper simple chunk object matching expected interface for _process_as_large
class SimpleChunk:
    def __init__(self, content, source="src", title="", metadata=None):
        self.content = content
        self.source = source
        self.title = title
        self.metadata = metadata or {}
        self.url = None
        self.created_at = None
        self.updated_at = None


def make_payload(**overrides) -> Payload:
    data = {
        "id": 1,
        "job_id": 2,
        "kb_type": "auto",
        "kb_external_id": "acme_project",
        "data": {"text": "Текст документа для обработки."},
        "external_url": "https://example.com/hook",
    }
    data.update(overrides)
    return Payload(**data)


def test_keys_to_json():
    """Every distinct key except content maps to a JSON pointer."""
    rows = [
        {"id": 1, "title": "First", "content": "text"},
        {"author": "Bob", "content": "more", "extra": 5},
        {"id": 2, "summary": "test"},
        "not a dict",
    ]
    expected = json.dumps(
        {
            "author": "/author",
            "extra": "/extra",
            "id": "/id",
            "summary": "/summary",
            "title": "/title",
        },
        ensure_ascii=False,
    )
    assert KnowledgePipeline._keys_to_json(rows) == expected


@pytest.mark.asyncio
async def test_process_as_small(stubbed_pipeline):
    docs = [
        RawElement(content="Doc one", source="src1"),
        RawElement(content="Doc two", source="src2"),
    ]

    result = await stubbed_pipeline._process_as_small(docs, total_tokens=1234)

    assert isinstance(result, list) and len(result) == 1
    item = result[0]
    # Verify that content matches the mocked enricher output
    assert item["content"] == '{"merged": true}'
    # Category should be knowledge_canvas
    assert item["category"] == DocCategory.KNOWLEDGE_CANVAS.value
    # Metadata should contain original document sources (deduped)
    assert set(item["metadata"]["original_documents"]) == {"src1", "src2"}
    assert item["metadata"]["total_original_tokens"] == 1234


@pytest.mark.asyncio
async def test_process_as_large_filters_nonuseful(stubbed_pipeline):
    raw_doc = RawElement(content="Some content", source="src")
    chunk1 = SimpleChunk(content="Useful part")
    chunk2 = SimpleChunk(content="Garbage")
    stubbed_pipeline.splitter.convert = Mock(return_value=[chunk1, chunk2])

    classified_useful = Classified(is_useful=True, category=DocCategory.FAQ, confidence=1.0)
    classified_useless = Classified(
        is_useful=False, category=DocCategory.DOCUMENT, confidence=0.5, reason="below_trash_size"
    )

    async def classify_side_effect(chunk):
        return classified_useful if chunk is chunk1 else classified_useless

    stubbed_pipeline.classifier.classify = AsyncMock(side_effect=classify_side_effect)
    stubbed_pipeline.faq_struct.extract = AsyncMock(
        return_value=[
            {
                "content": "FAQ item",
                "category": DocCategory.FAQ.value,
                "metadata": {},
                "source": "src",
                "title": "",
            }
        ]
    )
    stubbed_pipeline.glossary_struct.extract = AsyncMock(return_value=[])

    result = await stubbed_pipeline._process_as_large([raw_doc])

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["category"] == DocCategory.FAQ.value
    assert result[0]["content"] == "FAQ item"


# ------------------------------------------------------------------- branching


def test_select_branch_auto_follows_threshold(stubbed_pipeline):
    stubbed_pipeline.max_enrich_tokens = 1000
    assert stubbed_pipeline._select_branch("auto", 999) == "context"
    assert stubbed_pipeline._select_branch("auto", 1001) == "vector"


def test_select_branch_vector_is_pinned(stubbed_pipeline):
    stubbed_pipeline.max_enrich_tokens = 1000
    assert stubbed_pipeline._select_branch("vector", 1) == "vector"


def test_select_branch_context_rejects_oversized(stubbed_pipeline):
    stubbed_pipeline.max_enrich_tokens = 1000
    with pytest.raises(QuotaExceeded) as exc:
        stubbed_pipeline._select_branch("context", 5000)
    assert "5000" in str(exc.value)


@pytest.mark.asyncio
async def test_run_small_branch_never_touches_flowise(stubbed_pipeline, stub_documents):
    stub_documents(stubbed_pipeline, [RawElement(content="Короткий текст.", source="s")])

    result = await stubbed_pipeline.run(make_payload(kb_type="context"))

    assert result == 1
    stubbed_pipeline.flowise.document_store.find_document_store_by_name.assert_not_called()
    stubbed_pipeline.webhook_sender.send_payload.assert_awaited_once()
    sent = stubbed_pipeline.webhook_sender.send_payload.await_args[0][1]
    assert sent.kb_type == "context"
    assert sent.status == "success"


@pytest.mark.asyncio
async def test_run_large_branch_upserts_into_deterministic_store(stubbed_pipeline, stub_documents):
    from know_everything_ai.flowise.models import DocumentStore

    stub_documents(
        stubbed_pipeline,
        [RawElement(content="Короткий текст.", source="s")],
    )
    stubbed_pipeline.max_enrich_tokens = 1  # force the vector branch
    stubbed_pipeline.flowise.document_store.find_document_store_by_name = AsyncMock(return_value=None)
    stubbed_pipeline.flowise.document_store.create_document_store = AsyncMock(
        return_value=DocumentStore(id="store-1", name="kb_acme_project")
    )
    stubbed_pipeline.flowise.document_store.upsert_document = AsyncMock(return_value=Mock())
    stubbed_pipeline._process_as_large = AsyncMock(return_value=[{"content": "x", "category": "faq"}])

    result = await stubbed_pipeline.run(make_payload())

    assert result == 1
    created = stubbed_pipeline.flowise.document_store.create_document_store.await_args.kwargs["store"]
    assert created.name == "kb_acme_project"
    stubbed_pipeline.webhook_sender.send_payload.assert_awaited_once()
    sent = stubbed_pipeline.webhook_sender.send_payload.await_args[0][1]
    assert sent.kb_type == "vector"
    # text_out is the built-in registry id, not the Flowise store id; the store
    # id travels in its own nullable field so clients can keep both handles.
    assert sent.text_out == "1"
    assert sent.document_store_id == "store-1"
    # The fragment the structurizer produced was written to the built-in store.
    assert await stubbed_pipeline.stores.vector_store.count_chunks(1) == 1
    row = await stubbed_pipeline.stores.registry.get_by_external_id("acme_project")
    assert row["status"] == "ok"
    assert row["branch"] == "vector"
    assert row["document_store_id"] == "store-1"


@pytest.mark.asyncio
async def test_run_reports_cost_instead_of_empty_dict(stubbed_pipeline, stub_documents):
    stub_documents(stubbed_pipeline, [RawElement(content="Текст.", source="s")])

    await stubbed_pipeline.run(make_payload(kb_type="context"))

    sent = stubbed_pipeline.webhook_sender.send_payload.await_args[0][1]
    assert sent.cost["model_calls"] == 3
    assert sent.cost["model_total_tokens"] == 43
    assert sent.cost["by_component"]["classifier"]["calls"] == 2


@pytest.mark.asyncio
async def test_run_notifies_error_when_nothing_useful(stubbed_pipeline, stub_documents):
    stub_documents(stubbed_pipeline, [RawElement(content="Текст.", source="s")])
    stubbed_pipeline._process_as_small = AsyncMock(return_value=[])

    result = await stubbed_pipeline.run(make_payload(kb_type="context"))

    assert result == 0
    sent = stubbed_pipeline.webhook_sender.send_payload.await_args[0][1]
    assert sent.status == "error"
    assert sent.reason


# ---------------------------------------------------------------------- quotas


@pytest.mark.asyncio
async def test_too_many_files_is_rejected():
    kp = KnowledgePipeline(settings=Settings(_env_file=None, MAX_FILES_PER_KB=3))
    try:
        payload = make_payload(
            data={"files": {"txt": [f"https://example.com/{i}.txt" for i in range(5)]}}
        )
        with pytest.raises(QuotaExceeded) as exc:
            await kp._load_all_documents(payload)
        assert "3" in str(exc.value)
    finally:
        await kp.aclose()


@pytest.mark.asyncio
async def test_single_bad_file_does_not_sink_the_batch(stubbed_pipeline):
    """One unreadable file is logged and skipped; the rest still land."""
    good = [RawElement(content="ok", source="good.txt")]
    calls: list[str] = []

    async def transform(source):
        calls.append(source)
        if "bad" in source:
            raise ValueError("corrupt")
        return good

    loader = Mock(transform=AsyncMock(side_effect=transform))
    stubbed_pipeline._loaders["txt"] = loader

    payload = make_payload(
        data={"files": {"txt": ["https://example.com/bad.txt", "https://example.com/good.txt"]}}
    )

    assert await stubbed_pipeline._load_all_documents(payload) == good
    assert len(calls) == 2


def test_bare_string_payload_is_treated_as_the_document():
    assert KnowledgePipeline._parse_payload_data(make_payload(data="просто текст")) == {
        "text": "просто текст"
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/hook",
        "http://example.com/hook",
        # A webhook on an explicit port was rejected outright, which breaks
        # self-hosted buyers and any demo pointed at localhost:8099.
        "http://example.com:8099/hook",
        "https://hooks.example.com:8443/kb/result",
        "http://webhook-sink/hook",
        # Docker Compose service names carry underscores; rejecting them made
        # every in-network webhook look invalid.
        "http://webhook_sink/hook",
        "http://kb_webhook_sink:80/hook",
        "http://localhost:9000/callback",
    ],
)
def test_webhook_url_accepts_ports(url):
    payload = Payload.model_validate(
        {
            "id": 1,
            "job_id": 2,
            "kb_type": "auto",
            "kb_external_id": "acme_project",
            "data": "текст",
            "external_url": url,
        }
    )
    assert payload.external_url is not None


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/hook", "example.com/hook", "https://example.com:notaport/hook"],
)
def test_webhook_url_rejects_non_http(url):
    with pytest.raises(ValueError):
        Payload.model_validate(
            {
                "id": 1,
                "job_id": 2,
                "kb_type": "auto",
                "kb_external_id": "acme_project",
                "data": "текст",
                "external_url": url,
            }
        )


def test_parse_payload_data_treats_bare_url_as_a_file():
    """Payload documents "URL-строка" as accepted; it has to load the file."""
    payload = make_payload(
        data="http://storage.example.com/reports/contract.pdf",
    )
    parsed = KnowledgePipeline._parse_payload_data(payload)
    assert parsed == {
        "files": {"pdf": ["http://storage.example.com/reports/contract.pdf"]}
    }


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/a.PDF", "pdf"),
        ("https://example.com/a.docx", "docx"),
        ("https://example.com/a.xlsx", "xlsx"),
        ("https://example.com/a.csv", "csv"),
        ("https://example.com/a.html", "html"),
        ("https://example.com/a.md", "md"),
        # Unknown extension must not raise; the text loader handles bytes.
        ("https://example.com/download?id=7", "txt"),
    ],
)
def test_bare_url_maps_to_expected_loader(url, expected):
    parsed = KnowledgePipeline._parse_payload_data(make_payload(data=url))
    assert parsed["files"] == {expected: [url]}


def test_parse_payload_data_keeps_plain_text_as_text():
    payload = make_payload(data="Просто текст без ссылки на файл.")
    assert KnowledgePipeline._parse_payload_data(payload) == {
        "text": "Просто текст без ссылки на файл."
    }


def test_parse_payload_data_still_unwraps_json_envelope():
    payload = make_payload(
        data=json.dumps({"files": {"pdf": ["https://example.com/x.pdf"]}}),
    )
    assert KnowledgePipeline._parse_payload_data(payload) == {
        "files": {"pdf": ["https://example.com/x.pdf"]}
    }


@pytest.mark.asyncio
async def test_all_sources_failing_propagates_instead_of_empty_kb(pipeline, settings):
    """A provider outage must not be reported to the buyer as "nothing useful".

    Empty output is a terminal, acked answer, so swallowing the failure here
    turns a retryable 429 into a permanently empty knowledge base.
    """
    pipeline._loader_for = Mock(
        side_effect=lambda kind: Mock(
            transform=AsyncMock(side_effect=RuntimeError("429 insufficient balance"))
        )
    )
    payload = make_payload(
        kb_type="context",
        data={"files": {"pdf": ["https://example.com/a.pdf", "https://example.com/b.pdf"]}},
    )

    with pytest.raises(RuntimeError, match="429"):
        await pipeline._load_all_documents(payload)


@pytest.mark.asyncio
async def test_one_bad_source_still_yields_the_rest(pipeline, settings):
    """Partial failure stays tolerant: the readable documents must survive."""
    good = RawElement(content="Содержимое.", source="good", category=DocCategory.DOCUMENT)

    # transform() returns a list of elements; the first source reads fine and
    # the second raises.
    loader = Mock(transform=AsyncMock(side_effect=[[good], RuntimeError("boom")]))
    pipeline._loader_for = Mock(return_value=loader)
    payload = make_payload(
        kb_type="context",
        data={"files": {"pdf": ["https://example.com/a.pdf", "https://example.com/b.pdf"]}},
    )

    docs = await pipeline._load_all_documents(payload)
    assert [d.content for d in docs] == ["Содержимое."]


@pytest.mark.asyncio
async def test_prepare_does_not_call_any_llm(stubbed_pipeline, stub_documents):
    """Previewing must be free: the VLM only runs when the operator asks."""
    stub_documents(stubbed_pipeline, [RawElement(content="Короткий текст.", source="s")])
    stubbed_pipeline._process_as_small = AsyncMock(return_value=[{"content": "x"}])

    prepared = await stubbed_pipeline.prepare(make_payload(kb_type="context"))

    assert prepared.branch == "context"
    assert prepared.total_tokens > 0
    assert prepared.threshold > 0
    stubbed_pipeline._process_as_small.assert_not_awaited()
    stubbed_pipeline.enricher.create_knowledge_canvas.assert_not_called()


@pytest.mark.asyncio
async def test_run_preview_returns_items_without_pushing(stubbed_pipeline, stub_documents):
    """The UI shows the JSON first; nothing reaches Flowise or the webhook."""
    stub_documents(stubbed_pipeline, [RawElement(content="Короткий текст.", source="s")])
    stubbed_pipeline._process_as_small = AsyncMock(
        return_value=[{"content": "полотно", "category": "document"}]
    )

    result = await stubbed_pipeline.run_preview(make_payload(kb_type="context"))

    assert result.items == [{"content": "полотно", "category": "document"}]
    assert result.branch == "context"
    assert result.total_tokens > 0
    assert set(result.cost) >= {"total_tokens", "model_calls", "by_component"}

    stubbed_pipeline.flowise.document_store.find_document_store_by_name.assert_not_called()
    stubbed_pipeline.webhook_sender.send_payload.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_preview_uses_the_vector_branch_when_large(stubbed_pipeline, stub_documents):
    stub_documents(stubbed_pipeline, [RawElement(content="Текст.", source="s")])
    stubbed_pipeline._process_as_large = AsyncMock(return_value=[{"content": "чанк"}])

    result = await stubbed_pipeline.run_preview(
        make_payload(kb_type="vector")
    )

    assert result.branch == "vector"
    assert result.items == [{"content": "чанк"}]
    stubbed_pipeline._process_as_large.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_into_own_store_persists_vector_items(stubbed_pipeline):
    """The UI's write half lands the KB in the built-in store for the chat."""
    stubbed_pipeline.flowise = None  # the UI runs with FLOWISE_ENABLED=false
    payload = make_payload(kb_type="vector")
    items = [{"content": "QA", "category": "faq", "source": "s"}]

    kb_id = await stubbed_pipeline.store_into_own_store(
        payload,
        items,
        branch="vector",
        total_tokens=123,
        estimator="tiktoken",
        threshold=1000,
    )

    row = await stubbed_pipeline.stores.registry.get_by_external_id(
        payload.kb_external_id
    )
    assert row is not None and row["branch"] == "vector"
    assert row["item_count"] == 1
    assert await stubbed_pipeline.stores.vector_store.count_chunks(kb_id) == 1
    stubbed_pipeline.webhook_sender.send_payload.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_into_own_store_persists_context_canvas(stubbed_pipeline):
    stubbed_pipeline.flowise = None  # the UI runs with FLOWISE_ENABLED=false
    payload = make_payload(kb_type="context")
    items = [{"content": "Главный ответ", "category": "knowledge_canvas"}]

    kb_id = await stubbed_pipeline.store_into_own_store(
        payload,
        items,
        branch="context",
        total_tokens=50,
        estimator="tiktoken",
        threshold=1000,
    )

    row = await stubbed_pipeline.stores.registry.get_by_external_id(
        payload.kb_external_id
    )
    assert row is not None and row["branch"] == "context"
    # The canvas is kept for the read side and chunked for retrieval.
    assert row["item_count"] == 1
    assert row["canvas"] == items
    assert await stubbed_pipeline.stores.vector_store.count_chunks(kb_id) >= 1


# --------------------------------------------------------------------- append


@pytest.mark.asyncio
async def test_append_keeps_existing_fragments_and_grows_the_counters(stubbed_pipeline):
    """An update adds files; the ones already stored must survive it."""
    stubbed_pipeline.flowise = None
    payload = make_payload(kb_type="vector")

    kb_id = await stubbed_pipeline.store_into_own_store(
        payload,
        [{"content": "старый фрагмент", "category": "faq"}],
        branch="vector",
        total_tokens=100,
        estimator="tiktoken",
        threshold=1000,
    )
    appended_id = await stubbed_pipeline.store_into_own_store(
        payload,
        [{"content": "новый фрагмент", "category": "faq"}],
        branch="vector",
        total_tokens=50,
        estimator="tiktoken",
        threshold=1000,
        append=True,
    )

    assert appended_id == kb_id
    row = await stubbed_pipeline.stores.registry.get_by_external_id(
        payload.kb_external_id
    )
    # The row describes the whole base, not just the run that last touched it.
    assert row["item_count"] == 2
    assert row["total_tokens"] == 150
    contents = [hit.content for hit in await stubbed_pipeline.stores.vector_store.list_chunks(kb_id)]
    assert contents == ["старый фрагмент", "новый фрагмент"]


@pytest.mark.asyncio
async def test_append_merges_the_context_canvas(stubbed_pipeline):
    stubbed_pipeline.flowise = None
    payload = make_payload(kb_type="context")
    first = [{"content": "первое полотно", "category": "knowledge_canvas"}]
    second = [{"content": "второе полотно", "category": "knowledge_canvas"}]

    kb_id = await stubbed_pipeline.store_into_own_store(
        payload, first, branch="context",
        total_tokens=50, estimator="tiktoken", threshold=1000,
    )
    first_row = await stubbed_pipeline.stores.registry.get_by_external_id(
        payload.kb_external_id
    )
    first_chunks = await stubbed_pipeline.stores.vector_store.count_chunks(kb_id)

    await stubbed_pipeline.store_into_own_store(
        payload, second, branch="context",
        total_tokens=30, estimator="tiktoken", threshold=1000,
        append=True,
    )

    row = await stubbed_pipeline.stores.registry.get_by_external_id(
        payload.kb_external_id
    )
    assert row["canvas"] == first + second
    assert row["item_count"] == 2
    assert row["total_tokens"] == 80
    assert row["canvas_tokens"] > (first_row["canvas_tokens"] or 0)
    assert await stubbed_pipeline.stores.vector_store.count_chunks(kb_id) > first_chunks


@pytest.mark.asyncio
async def test_append_rejects_a_base_that_does_not_exist(stubbed_pipeline):
    """Appending into nothing would create a base no one asked to create."""
    stubbed_pipeline.flowise = None

    with pytest.raises(ValueError, match="does not exist"):
        await stubbed_pipeline.store_into_own_store(
            make_payload(),
            [{"content": "x"}],
            branch="vector",
            total_tokens=1,
            estimator="t",
            threshold=1,
            append=True,
        )


@pytest.mark.asyncio
async def test_append_rejects_a_branch_mismatch(stubbed_pipeline):
    """A context canvas cannot be spliced into a vector base and vice versa."""
    stubbed_pipeline.flowise = None
    payload = make_payload(kb_type="vector")
    await stubbed_pipeline.store_into_own_store(
        payload, [{"content": "старый"}],
        branch="vector", total_tokens=10, estimator="t", threshold=100,
    )

    with pytest.raises(ValueError, match="incompatible"):
        await stubbed_pipeline.store_into_own_store(
            payload, [{"content": "новый"}],
            branch="context", total_tokens=10, estimator="t", threshold=100,
            append=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("append", "expected_cleanup"),
    [(True, "none"), (False, "full")],
)
async def test_push_to_store_passes_the_cleanup_mode_on(
    stubbed_pipeline, append: bool, expected_cleanup: str
):
    """Appending must not let the record manager clean up what it does not see.

    The documents outside this batch are exactly the ones being kept, so an
    append runs with cleanup off while the normal push keeps the mode the
    worker has always run with.
    """
    from know_everything_ai.flowise.models import DocumentStore

    stubbed_pipeline.flowise.document_store.find_document_store_by_name = AsyncMock(
        return_value=DocumentStore(id="store-1", name="kb_acme_project")
    )
    stubbed_pipeline.flowise.document_store.upsert_document = AsyncMock(
        return_value=Mock()
    )

    await stubbed_pipeline.push_to_store(
        make_payload(), [{"content": "x"}], append=append
    )

    config = stubbed_pipeline.flowise.document_store.upsert_document.await_args.kwargs[
        "config"
    ]
    assert config.recordManager.config["cleanupMode"] == expected_cleanup


def test_bare_local_file_path_is_routed_to_a_loader(tmp_path):
    """An uploaded document is a path, and it used to become its own content."""
    source = tmp_path / "contract.txt"
    source.write_text("Статья 1. Предмет договора.", encoding="utf-8")

    parsed = KnowledgePipeline._parse_payload_data(make_payload(data=str(source)))

    assert parsed == {"files": {"txt": [str(source)]}}


def test_bare_local_path_picks_the_loader_from_the_extension(tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")

    parsed = KnowledgePipeline._parse_payload_data(make_payload(data=str(source)))

    assert parsed == {"files": {"pdf": [str(source)]}}


def test_bare_string_that_is_not_a_file_stays_text():
    """Pasted prose ending in .txt must not be treated as a path."""
    parsed = KnowledgePipeline._parse_payload_data(make_payload(data="см. notes.txt"))

    assert parsed == {"text": "см. notes.txt"}


def test_list_of_local_paths_reaches_the_text_loader():
    parsed = KnowledgePipeline._parse_payload_data(
        make_payload(data=["/app/data/a.txt", "/app/data/b.txt"])
    )

    assert parsed == {"files": {"txt": ["/app/data/a.txt", "/app/data/b.txt"]}}
