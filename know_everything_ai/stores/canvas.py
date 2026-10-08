"""Split a knowledge canvas into fragments that can be embedded.

The canvas is one blob of text produced for the small-document branch, so there
is nothing to retrieve from it unless it is cut into pieces first. It is also
not Markdown despite living in a Markdown-shaped pipeline: the enricher returns a
fenced JSON object (see ``KNOWLEDGE_CANVAS_PROMPT``), so cutting it on headings
would saw through glossary definitions mid-sentence.

Sections therefore come from the JSON structure — one fragment per top-level
key, one per element of a list of objects — which keeps an entry and its
definition together. Only when the payload is not parseable does it fall back to
treating the whole thing as a single fragment, so a malformed canvas degrades
instead of failing the job.
"""

from __future__ import annotations

import json
import re
from typing import Any

from know_everything_ai.schemas import DocCategory, RawElement
from know_everything_ai.stores.base import Chunk
from know_everything_ai.utils.token_counter import TokenEstimator

# The enricher is told to emit a single well-formed JSON object and models like
# to wrap it in a fence anyway. A fence left in the text becomes part of what is
# embedded and shows up in the retrieved excerpt.
_FENCE_RE = re.compile(
    r"\A```[A-Za-z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?```\s*\Z",
    re.DOTALL,
)


def strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group("body").strip() if match else text.strip()


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    # Compact: indentation is for humans and costs embedding budget.
    return json.dumps(value, ensure_ascii=False)


def canvas_sections(content: str) -> list[tuple[str, str]]:
    """Return ``(section_title, section_text)`` pairs for a canvas body.

    A top-level key holding a list of objects becomes one section per element
    rather than one long section — that is the granularity a question about a
    single glossary term can hit.
    """
    body = strip_fence(content)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return [("", body)]

    if isinstance(parsed, dict):
        sections: list[tuple[str, str]] = []
        for key, value in parsed.items():
            if isinstance(value, list) and value:
                if all(isinstance(element, dict) for element in value):
                    sections.extend(
                        (f"{key} / {index + 1}", _render(element))
                        for index, element in enumerate(value)
                    )
                else:
                    sections.append((str(key), _render(value)))
            else:
                sections.append((str(key), _render(value)))
        # An object with only empty members would otherwise embed as nothing.
        return [(title, text) for title, text in sections if text.strip()] or [
            ("", body)
        ]

    if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
        return [(f"{index + 1}", _render(element)) for index, element in enumerate(parsed)]

    return [("", body)]


def split_oversized(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    estimator: TokenEstimator,
) -> list[str]:
    """Cut one section down to the embedding window, keeping an overlap.

    Sizes are measured once against the whole section rather than per candidate
    window: re-running the tokenizer for every window makes splitting quadratic
    in the length of the text for no gain in accuracy. A section that already
    fits — the common case — costs a single estimate.
    """
    total = estimator.estimate(text)
    if total <= max_tokens or max_tokens <= 0:
        return [text]

    chars_per_token = max(1.0, len(text) / total)
    window = max(1, int(max_tokens * chars_per_token))
    overlap = min(window - 1, max(0, int(overlap_tokens * chars_per_token)))

    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + window)
        if end < length:
            # Prefer a word boundary over a token split; a fragment cut in the
            # middle of a word embeds as two halves of neither.
            boundary = text.rfind(" ", start + window // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        # +1 guarantees progress even when the overlap collapses onto the start.
        start = max(end - overlap, start + 1)
    return pieces


def canvas_to_chunks(
    items: list[RawElement] | list[dict[str, Any]],
    *,
    max_tokens: int,
    overlap_tokens: int,
    estimator: TokenEstimator,
) -> list[Chunk]:
    """Turn processed context-branch output into embeddable fragments.

    ``items`` carries the same dicts the webhook receives, so the stored
    fragments are exactly what the caller was shown — not a re-derivation that
    could disagree with it.
    """
    chunks: list[Chunk] = []
    for item in items:
        # Loosely typed on purpose: the same function serves the pydantic model
        # and the plain dict the webhook receives, and the store wants both.
        content: str
        raw_category: Any
        source: Any
        title: Any
        metadata: dict[str, Any]
        if isinstance(item, RawElement):
            content = item.content
            raw_category = item.category
            source = item.source
            title = item.title
            metadata = dict(item.metadata)
        else:
            content = str(item.get("content", ""))
            raw_category = item.get("category")
            source = item.get("source")
            title = item.get("title")
            metadata = dict(item.get("metadata") or {})

        if not content.strip():
            continue

        # RawElement carries the enum, the webhook-shaped dict carries the wire
        # string. The column stores the string, and leaving an enum there would
        # serialize it as "DocCategory.TABLE" downstream.
        if isinstance(raw_category, DocCategory):
            raw_category = raw_category.value
        category = raw_category or DocCategory.KNOWLEDGE_CANVAS.value

        for section_title, section_text in canvas_sections(content):
            for piece in split_oversized(
                section_text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                estimator=estimator,
            ):
                piece_metadata = dict(metadata)
                if section_title:
                    piece_metadata["section"] = section_title
                combined_title = (
                    f"{title} · {section_title}"
                    if title and section_title
                    else (title or section_title or "")
                )
                chunks.append(
                    Chunk(
                        content=piece,
                        category=category,
                        source=source,
                        title=combined_title,
                        metadata=piece_metadata,
                    )
                )
    return chunks
