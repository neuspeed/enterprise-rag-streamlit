"""Bytes-to-text decoding for fetched sources.

Shared by every loader that ends up with raw bytes. Russian documents arrive as
cp1251, koi8-r and windows-1252 about as often as UTF-8, and decoding them as
UTF-8 with ``errors="replace"`` turns a page of text into a page of U+FFFD —
which then gets embedded and stored as if it were the source.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger("decoding")

#: Tried in order when neither the declared nor the detected charset works.
FALLBACK_ENCODINGS = ("utf-8", "cp1251", "koi8-r", "windows-1252", "latin-1")


def decode_bytes(payload: bytes, content_type: str | None = None) -> str:
    """Decode ``payload``, preferring its declared charset, then detection."""
    declared = _charset_from_content_type(content_type)
    if declared:
        decoded = _try(payload, declared)
        if decoded is not None:
            return decoded

    detected = _detect(payload)
    if detected:
        decoded = _try(payload, detected)
        if decoded is not None:
            return decoded

    for encoding in FALLBACK_ENCODINGS:
        decoded = _try(payload, encoding)
        if decoded is not None:
            return decoded
    return payload.decode("utf-8", errors="replace")


def _charset_from_content_type(content_type: str | None) -> str | None:
    if not content_type or "charset=" not in content_type:
        return None
    return content_type.split("charset=", 1)[1].split(";")[0].strip().strip('"')


def _detect(payload: bytes) -> str | None:
    try:
        import chardet
    except ImportError:  # pragma: no cover - chardet is a hard dependency
        return None
    result = chardet.detect(payload) or {}
    encoding = result.get("encoding")
    confidence = result.get("confidence") or 0.0
    if encoding and confidence > 0.5:
        return encoding
    return None


def _try(payload: bytes, encoding: str) -> str | None:
    try:
        text = payload.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        return None
    # A wrong single-byte codec decodes without error and produces mojibake.
    # A high share of U+FFFD is the tell that we guessed wrong.
    if text.count("�") > len(text) * 0.01:
        return None
    return text
