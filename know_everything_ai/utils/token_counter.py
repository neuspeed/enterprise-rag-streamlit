"""Token counting.

The Small/Large branch decision is made by comparing a measured token count
against ``MAX_CKB_TOKENS``, so a wrong estimate does not merely report a wrong
number: it picks the wrong branch, and a document the enricher cannot hold gets
sent to it anyway and overflows the model's context window.

The inherited estimator assumed 3.2 characters per token for Cyrillic. Measured
against ``cl100k_base``, Russian prose runs at 2.35 — the heuristic under-counted
by 38%, so a document it scored at 200k was really 280k.

The heuristic blends the Russian and English coefficients by the *share* of
Cyrillic letters instead of switching on a boolean, because a knowledge base
mixing Russian prose with JSON keys and CLI flags is the common case, not an
edge case. Measured against ``cl100k_base`` on this repository's sources:

* Russian prose: within 1% (the 800k-character corpus lands at 0.7%).
* Code and mixed content: ~8% mean error, ~24% worst case.

That spread is inherent to a characters-per-token model: prose tokenises at
2.4 chars/token and source code at 2.0, and no single coefficient covers both.
A two-feature fit on language and whitespace share cuts the mean to 6.6% under
leave-one-out but does not improve the worst case, and costs numpy plus three
constants fitted to one codebase — so it was not taken. Install ``tiktoken``
(the default) and this band does not apply; the heuristic is the fallback for
deployments without it, and ``active_backend`` says which one produced a
number.

Three backends:

``tiktoken``
    Exact for the configured encoding. Always preferred.
``auto``
    ``tiktoken`` when the configured model maps to a known OpenAI encoding,
    otherwise the heuristic. The default: a buyer pointing at vLLM with a local
    model name still gets a sane number instead of an exception.
``heuristic``
    The calibrated character ratio below.

``TokenEstimator.active_backend`` reports which one produced a number, so the
figure shown to a buyer can be attributed rather than trusted blindly.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from know_everything_ai.settings import Settings

log = structlog.get_logger("token_counter")

_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_WORD_BREAK_RE = re.compile(r"[ \n\t]")

#: Below this length, "no whitespace" is just a short string, not a payload.
_DENSE_MIN_CHARS = 400
#: Whitespace share below which text is treated as encoded data.
_DENSE_WHITESPACE_RATIO = 0.01
#: Characters per token for encoded payloads. Measured against cl100k_base:
#: base64 1.40, hex 1.78, prose 2.42. The lowest wins so the estimate errs high,
#: which is the safe direction for a branch threshold.
DENSE_CHARS_PER_TOKEN = 1.4

#: Cached ``tiktoken`` encodings. Building one downloads and parses a BPE table,
#: so this must not happen per call.
_ENCODINGS: dict[str, Any] = {}

_tiktoken: Any | None = None  # tiktoken ships no usable stubs
try:  # pragma: no cover - exercised by whichever backend is installed
    import tiktoken as _tiktoken
except ImportError:  # pragma: no cover
    _tiktoken = None


def tiktoken_available() -> bool:
    return _tiktoken is not None


def _load_encoding(name: str):
    """Return a cached ``tiktoken`` encoding, or None if unavailable."""
    if _tiktoken is None:
        return None
    if name not in _ENCODINGS:
        try:
            _ENCODINGS[name] = _tiktoken.get_encoding(name)
        except Exception as exc:  # unknown encoding name, missing vocab, ...
            log.warning("tokenizer_encoding_unavailable", encoding=name, error=str(exc))
            _ENCODINGS[name] = None
    return _ENCODINGS[name]


def _encoding_for_model(model_name: str):
    """Return the encoding OpenAI uses for ``model_name``, or None."""
    if _tiktoken is None or not model_name:
        return None
    try:
        return _tiktoken.encoding_for_model(model_name)
    except Exception:
        # Not an OpenAI model — expected when the buyer runs a local model.
        return None


class TokenEstimator:
    """Counts tokens according to the configured backend."""

    def __init__(self, settings: Settings) -> None:
        self.backend = settings.TOKENIZER_BACKEND
        self.encoding_name = settings.TOKENIZER_ENCODING or "cl100k_base"
        self.ru_chars_per_token = settings.TOKENIZER_RU_CHARS_PER_TOKEN
        self.en_chars_per_token = settings.TOKENIZER_EN_CHARS_PER_TOKEN
        self.dense_chars_per_token = DENSE_CHARS_PER_TOKEN
        self._encoding = None
        self._encoding_loaded = False

    def _resolve_encoding(self, model_name: str | None):
        if self._encoding_loaded:
            return self._encoding
        self._encoding_loaded = True

        if self.backend == "heuristic" or not tiktoken_available():
            self._encoding = None
        elif self.backend == "tiktoken":
            self._encoding = _load_encoding(self.encoding_name)
        else:  # auto: prefer the model's own encoding, fall back to configured
            self._encoding = _encoding_for_model(model_name or "") or _load_encoding(
                self.encoding_name
            )
        return self._encoding

    def active_backend(self, model_name: str | None = None) -> str:
        """Which estimator a call with this model would actually use."""
        if self._resolve_encoding(model_name) is not None:
            return "tiktoken"
        return "heuristic"

    def estimate(self, text: str, model_name: str | None = None) -> int:
        if not text:
            return 0

        encoding = self._resolve_encoding(model_name)
        if encoding is not None:
            return len(encoding.encode(text))
        return self._heuristic(text)

    def _heuristic(self, text: str) -> int:
        if self._is_dense_payload(text):
            return max(1, round(len(text) / self.dense_chars_per_token))

        cyrillic = _CYRILLIC_RE.findall(text)
        latin = _LATIN_RE.findall(text)
        cased = len(cyrillic) + len(latin)
        if cased == 0:
            # Punctuation, whitespace and digits only: nothing to calibrate on.
            return max(1, round(len(text) / self.dense_chars_per_token))

        ru_share = len(cyrillic) / cased
        chars_per_token = (
            self.en_chars_per_token
            + (self.ru_chars_per_token - self.en_chars_per_token) * ru_share
        )
        return max(1, round(len(text) / chars_per_token))

    @staticmethod
    def _is_dense_payload(text: str) -> bool:
        """Whether ``text`` looks like encoded data rather than prose.

        Base64, hex dumps and minified assets are full of cased letters, so
        "count the Cyrillic" alone does not catch them: a base64 attachment
        scored at 384 tokens against a true 1240 landed three times under the
        limit. Prose, markup and JSON all separate words with whitespace;
        a long run with almost none of it is a payload.
        """
        if len(text) < _DENSE_MIN_CHARS:
            return False
        if not _WORD_BREAK_RE.search(text):
            return True
        whitespace = sum(text.count(sep) for sep in (" ", "\n", "\t"))
        return whitespace / len(text) < _DENSE_WHITESPACE_RATIO


def get_estimator(settings: Settings) -> TokenEstimator:
    return TokenEstimator(settings)


def estimate_tokens(
    text: str,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> int:
    """Count tokens in ``text``.

    ``settings`` selects the backend and the calibrated coefficients. Passing it
    is what makes the number trustworthy; without it a standalone default
    estimator is used, which is adequate for tests and ad-hoc counting.
    """
    return get_estimator(settings or Settings()).estimate(text, model_name)
