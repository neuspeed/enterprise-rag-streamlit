"""Loader construction tests.

These loaders are wired together by keyword in ``KnowledgePipeline.__init__``,
where a mistyped argument surfaces only when a knowledge base actually contains
that file type. ``usage_name`` was passed to ``PDFParser`` without being
forwarded, so every PDF raised ``TypeError`` at construction — and the pipeline
tests missed it because the vector branch mocks the loader out.

These tests construct every loader for real, offline.
"""

from __future__ import annotations

import pytest

from know_everything_ai.loaders.registry import LOADER_TYPES, create_loader
from know_everything_ai.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        ENRICHER_API_URL="",
        ENRICHER_MODEL_NAME="",
        CLASSIFIER_API_URL="",
        CLASSIFIER_MODEL_NAME="",
        VLM_API_URL="",
        VLM_MODEL_NAME="",
        FLOWISE_ENABLED=False,
    )


@pytest.mark.parametrize("source_type", sorted(LOADER_TYPES))
def test_every_registered_loader_constructs(source_type, settings):
    loader = create_loader(source_type, settings)
    assert loader.settings is settings


def test_planned_formats_are_all_supported():
    """Formats named in the product description must resolve to a loader."""
    for source_type in ("pdf", "docx", "xlsx", "csv", "txt", "html"):
        assert source_type in LOADER_TYPES, source_type


def test_pdf_loader_exposes_its_vlm_for_billing(settings):
    """A 200-page VLM parse costs more than the enrichment pass it feeds.

    If the parser is not reachable through ``llm_components`` its tokens are
    never billed, because the pipeline can only sum the components it can see.
    """
    loader = create_loader("pdf", settings)
    components = loader.llm_components()
    assert len(components) == 1
    assert components[0].usage_name == "vlm"


def test_non_llm_loaders_bill_nothing(settings):
    for source_type in ("txt", "html", "csv"):
        assert create_loader(source_type, settings).llm_components() == ()


def test_loaders_without_a_model_report_zero_usage(settings):
    loader = create_loader("txt", settings)
    assert loader.llm_components() == ()


def test_registry_rejects_unknown_types(settings):
    from know_everything_ai.utils.source_client import SourceFetchError

    with pytest.raises(SourceFetchError):
        create_loader("exe", settings)


def test_dot_prefixed_type_is_accepted(settings):
    """Payload keys arrive as file extensions, sometimes with the dot."""
    assert create_loader(".txt", settings) is not None


def test_type_is_case_insensitive(settings):
    assert type(create_loader("PDF", settings)) is type(create_loader("pdf", settings))
