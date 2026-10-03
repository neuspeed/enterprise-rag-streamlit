import json
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.pipeline import KnowledgePipeline
from app.utils.schemas import RawElement, DocCategory, Classified

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

@pytest.fixture
def pipeline(monkeypatch):
    # Instantiate with real Settings (env values are harmless)
    from app.settings import Settings
    kp = KnowledgePipeline(settings=Settings())
    # Replace heavy dependencies with mocks
    kp.enricher = Mock()
    kp.enricher.create_knowledge_canvas = AsyncMock(return_value="{\"merged\": true}")
    kp.chunk_merger = Mock()
    kp.chunk_merger.merge = Mock(side_effect=lambda docs: docs)  # identity
    kp.splitter = Mock()
    kp.classifier = Mock()
    kp.faq_struct = Mock()
    kp.glossary_struct = Mock()
    # Mock flowise and webhook sender to avoid network calls
    kp.flowise = Mock()
    kp.webhook_sender = Mock()
    return kp

def test_keys_to_json(pipeline):
    """Directly test the private __keys_to_json helper."""
    input_data = [
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
        indent=2,
    )
    # Call the mangled private method on the pipeline instance
    result = pipeline._KnowledgePipeline__keys_to_json(input_data)
    assert result == expected

@pytest.mark.asyncio
async def test_process_as_small(pipeline):
    # Prepare two RawElement docs
    docs = [
        RawElement(content="Doc one", source="src1"),
        RawElement(content="Doc two", source="src2"),
    ]
    # Call the private method (name‑mangled)
    result = await pipeline._process_as_small(docs)
    assert isinstance(result, list) and len(result) == 1
    item = result[0]
    # Verify that content matches the mocked enricher output
    assert item["content"] == "{\"merged\": true}"
    # Category should be knowledge_canvas
    assert item["category"] == DocCategory.KNOWLEDGE_CANVAS.value
    # Metadata should contain original document sources (deduped)
    sources = item["metadata"]["original_documents"]
    assert set(sources) == {"src1", "src2"}
    # Total token estimate stub (estimate_tokens is imported, we trust its behavior)
    assert isinstance(item["metadata"]["total_original_tokens"], int)

@pytest.mark.asyncio
async def test_process_as_large_filters_nonuseful(pipeline, monkeypatch):
    # One raw doc -> splitter returns two chunks
    raw_doc = RawElement(content="Some content", source="src")
    chunk1 = SimpleChunk(content="Useful part")
    chunk2 = SimpleChunk(content="Garbage")
    pipeline.splitter.convert = Mock(return_value=[chunk1, chunk2])
    # Classifier: first chunk useful, second not useful
    classified_useful = Classified(is_useful=True, category=DocCategory.FAQ, confidence=1.0)
    classified_useless = Classified(is_useful=False, category=DocCategory.DOCUMENT, confidence=0.5)
    async def classify_side_effect(chunk):
        return classified_useful if chunk is chunk1 else classified_useless
    pipeline.classifier.classify = AsyncMock(side_effect=classify_side_effect)
    # FAQ and glossary extracts return simple lists
    pipeline.faq_struct.extract = AsyncMock(return_value=[{"content": "FAQ item", "category": DocCategory.FAQ.value, "metadata": {}, "source": "src", "title": ""}])
    pipeline.glossary_struct.extract = AsyncMock(return_value=[])
    # Execute
    result = await pipeline._process_as_large([raw_doc])
    # Should contain only the FAQ item from the useful chunk
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["category"] == DocCategory.FAQ.value
    assert result[0]["content"] == "FAQ item"
