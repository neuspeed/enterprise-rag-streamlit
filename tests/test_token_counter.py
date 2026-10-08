"""Tokenizer tests.

These are the numbers the Small/Large branch decision rests on. A regression
here does not show up as a wrong count on a dashboard; it silently routes
documents to a branch that cannot hold them, so the thresholds are pinned
against a real encoding rather than against the implementation's own output.
"""

from __future__ import annotations

import pytest

from know_everything_ai.settings import Settings
from know_everything_ai.utils.token_counter import (
    TokenEstimator,
    tiktoken_available,
)

pytestmark = pytest.mark.skipif(
    not tiktoken_available(), reason="tiktoken is not installed"
)

RUSSIAN = (
    "Договор поставки заключается между сторонами и определяет порядок передачи "
    "товара, его оплаты и приёмки. Срок действия договора составляет двенадцать "
    "месяцев с даты подписания, если иное не предусмотрено приложением."
)
ENGLISH = (
    "The supply agreement is concluded between the parties and governs delivery, "
    "payment and acceptance of the goods. The agreement remains in force for "
    "twelve months from the date of signature unless otherwise provided."
)


def estimator(**overrides) -> TokenEstimator:
    base = {"_env_file": None, "TOKENIZER_BACKEND": "tiktoken"}
    base.update(overrides)
    return TokenEstimator(Settings(**base))


def test_tiktoken_backend_is_exact():
    """With tiktoken selected, the count must equal the encoding's own count."""
    import tiktoken

    est = estimator()
    assert est.active_backend() == "tiktoken"
    encoding = tiktoken.get_encoding("cl100k_base")
    assert est.estimate(RUSSIAN) == len(encoding.encode(RUSSIAN))


def test_empty_string_costs_nothing():
    assert estimator().estimate("") == 0


def test_russian_runs_at_about_2_35_chars_per_token():
    """The calibrated coefficient is the reason this module was rewritten.

    The inherited 3.2 under-counted Russian prose by ~38%, which meant a
    document genuinely at 280k tokens was scored at 200k and pushed down the
    enricher branch that could not hold it.
    """
    import tiktoken

    text = RUSSIAN * 10
    truth = len(tiktoken.get_encoding("cl100k_base").encode(text))
    est = estimator(TOKENIZER_BACKEND="heuristic")
    assert est.active_backend() == "heuristic"

    estimate = est.estimate(text)
    assert abs(estimate - truth) / truth < 0.05


def test_heuristic_is_far_closer_than_the_inherited_ratio():
    """Guard against a silent return to 3.2 chars/token for Cyrillic."""
    import tiktoken

    text = RUSSIAN * 10
    truth = len(tiktoken.get_encoding("cl100k_base").encode(text))
    est = estimator(TOKENIZER_BACKEND="heuristic")

    inherited = round(len(text) / 3.2)
    assert abs(est.estimate(text) - truth) < abs(inherited - truth)


def test_mixed_text_is_interpolated_not_bucketed():
    """A knowledge base mixes Russian prose with JSON keys and CLI flags."""
    est = estimator(TOKENIZER_BACKEND="heuristic")
    russian_only = est.estimate(RUSSIAN * 4)
    english_only = est.estimate(ENGLISH * 4)
    mixed = est.estimate(RUSSIAN * 4 + ENGLISH * 4)

    # Russian is denser than English, so it costs more tokens for the same
    # character count; the mixed figure must land between the two.
    assert english_only < russian_only
    assert russian_only < mixed < english_only + russian_only


def test_content_free_blobs_are_not_treated_as_prose():
    """Base64 and hex dumps tokenise several times denser than text."""
    import tiktoken

    blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5" * 40
    truth = len(tiktoken.get_encoding("cl100k_base").encode(blob))
    estimate = estimator(TOKENIZER_BACKEND="heuristic").estimate(blob)

    assert estimate >= truth * 0.8


def test_auto_backend_falls_back_without_raising():
    """A buyer pointing at vLLM with an unknown local model must still get a count."""
    est = estimator(TOKENIZER_BACKEND="auto")
    assert est.estimate(RUSSIAN) > 0


def test_encoding_is_resolved_once():
    """Building a BPE table downloads and parses; it must be cached."""
    est = estimator()
    assert est.estimate(RUSSIAN) > 0
    assert est._resolve_encoding("gpt-4") is est._resolve_encoding("gpt-4")
