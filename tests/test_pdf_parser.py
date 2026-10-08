"""How the PDF parser reads a vision model's reply.

The parser's only hard contract with the model is that the reply is JSON. Vision
models are not reliable about that, and the failures are not the model's fault:
the transcription is right, the envelope is wrong. These tests pin the envelope
tolerance that real runs needed.
"""

from __future__ import annotations

import json

import pytest

from know_everything_ai.parsers.pdf_parser import _batched, _loads

# Captured from a real glm-4.6v-flash reply that the parser rejected: correct
# transcription, but a raw newline inside a JSON string value instead of the
# escaped form. Strict json.loads calls this invalid.
RAW_NEWLINE_IN_STRING = (
    '[\n  {\n    "content": "Договор поставки № 41-А\n'
    'г. Москва, 17 марта 2026 г.\nПоставщик: ООО «Северный Путь»",\n'
    '    "category": "document_chunk",\n'
    '    "title": "Договор",\n'
    '    "metadata": {}\n  }\n]'
)

FENCED = (
    "```json\n"
    '[\n  {"content": "Раздел 1", "category": "document_chunk",'
    ' "title": "Раздел 1", "metadata": {}}\n]\n'
    "```"
)


def test_raw_newline_inside_a_string_still_parses() -> None:
    """The regression: right content, invalid envelope, page lost as a result."""
    chunks = _loads(RAW_NEWLINE_IN_STRING)

    assert chunks is not None, "a transcribed page must not be dropped for whitespace"
    assert chunks[0]["content"].startswith("Договор поставки")
    assert "\n" in chunks[0]["content"], "the newline is content, not corruption"


def test_strict_json_rejects_the_same_text() -> None:
    """Pins why strict=False is needed, so nobody 'simplifies' it away."""
    with pytest.raises(json.JSONDecodeError):
        json.loads(RAW_NEWLINE_IN_STRING, strict=True)


def test_fenced_json_is_accepted() -> None:
    chunks = _loads(FENCED)

    assert chunks is not None
    assert chunks[0]["content"] == "Раздел 1"


def test_a_single_object_is_wrapped_in_a_list() -> None:
    chunks = _loads('{"content": "Термин", "category": "glossary"}')

    assert chunks == [{"content": "Термин", "category": "glossary"}]


def test_truncated_json_is_reported_as_unparseable() -> None:
    """An output cap truncates the array; that is a real failure, not whitespace."""
    assert _loads('[{"content": "начало') is None


def test_prose_instead_of_json_is_reported_as_unparseable() -> None:
    assert _loads("Извините, я не могу прочитать этот документ.") is None


def test_empty_reply_is_reported_as_unparseable() -> None:
    assert _loads("") is None
    assert _loads("   \n  ") is None


def test_batching_covers_every_item_exactly_once() -> None:
    items = list(range(7))

    batches = list(_batched(items, 3))

    assert batches == [[0, 1, 2], [3, 4, 5], [6]]
    assert [item for batch in batches for item in batch] == items


def test_batching_an_empty_input_yields_nothing() -> None:
    assert list(_batched([], 3)) == []
