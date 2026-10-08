"""Shared test fixtures.

Two things live here. First, the ``settings`` fixture: every component reads
:class:`Settings`, which in production is populated from the environment, so
tests must pin the values they depend on instead of inheriting whatever the
developer's shell happens to export. Second, ``pipeline``: building a real
:class:`KnowledgePipeline` allocates an httpx pool, and leaking one per test
leaks one socket per test for the whole session.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from know_everything_ai.pipeline import KnowledgePipeline
from know_everything_ai.settings import Settings
from know_everything_ai.stores.memory import MemoryStoreContext
from know_everything_ai.utils.logging import configure_logging

# Windows defaults to ProactorEventLoop; psycopg's async mode rejects it. Set
# the selector policy before pytest-asyncio creates any loop so that tests which
# do touch the database can on the host that develops on them.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#: Fixed token counts so cost assertions do not depend on a live model.
USAGE_ENRICHER = {
    "calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
}
USAGE_CLASSIFIER = {
    "calls": 2, "prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28
}


@pytest.fixture(autouse=True, scope="session")
def _logging() -> None:
    """Wire structlog into stdlib so ``caplog`` can see emitted records.

    The classifier used to call ``structlog.configure`` at import time purely so
    that caplog would catch its output. Configuring once, here, tests the same
    path production uses instead of a special case that only existed for tests.
    """
    configure_logging("DEBUG", json_output=False)


@pytest.fixture
def settings() -> Settings:
    """Settings with no external dependency configured."""
    return Settings(
        _env_file=None,
        ENRICHER_API_URL="",
        ENRICHER_API_KEY="",
        ENRICHER_MODEL_NAME="",
        CLASSIFIER_API_URL="",
        CLASSIFIER_API_KEY="",
        CLASSIFIER_MODEL_NAME="",
        VLM_API_URL="",
        VLM_MODEL_NAME="",
        FLOWISE_ENABLED=False,
    )


@pytest_asyncio.fixture
async def pipeline(settings: Settings):
    """A real pipeline with nothing external wired up, for lifecycle tests."""
    kp = KnowledgePipeline(settings=settings)
    try:
        yield kp
    finally:
        await kp.aclose()


@pytest_asyncio.fixture
async def stubbed_pipeline(settings: Settings):
    """Pipeline with enrichment, classification and outbound calls mocked.

    Yields the object itself rather than a factory because every test that needs
    it also needs to inspect what it was asked to send.
    """
    kp = KnowledgePipeline(settings=settings)
    # Persistence is exercised against the in-memory store, not the database:
    # the ordering contract is what matters here, and the real store is covered
    # by the integration gate instead.
    kp.stores = MemoryStoreContext(settings)
    kp.enricher = Mock()
    kp.enricher.create_knowledge_canvas = AsyncMock(return_value='{"merged": true}')
    kp.enricher.usage_name = "enricher"
    kp.enricher.usage = dict(USAGE_ENRICHER)
    kp.chunk_merger = Mock()
    kp.chunk_merger.merge = Mock(side_effect=list)  # identity
    kp.splitter = Mock()
    kp.classifier = Mock()
    kp.classifier.usage_name = "classifier"
    kp.classifier.usage = dict(USAGE_CLASSIFIER)
    kp.faq_struct = Mock()
    kp.glossary_struct = Mock()
    kp.flowise = AsyncMock()
    kp.webhook_sender = Mock()
    kp.webhook_sender.send_payload = AsyncMock(return_value=True)
    try:
        yield kp
    finally:
        await kp.aclose()


@pytest.fixture
def stub_documents():
    """Replace document loading so a test can drive the branch logic directly."""

    def _stub(kp, docs):
        kp._load_all_documents = AsyncMock(return_value=docs)

    return _stub
