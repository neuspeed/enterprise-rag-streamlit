"""Unit tests for the answer agent.

``complete_stream`` is exercised against fake OpenAI clients so failover across
targets can be asserted without any network; the stream-event orchestrator runs
against a fake agent so the sources/delta/done contract is pinned independent of
any particular model backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from know_everything_ai.agent.answer import (
    CANNED_NO_HITS,
    build_answer_user_message,
    stream_answer_events,
    usage_delta,
)
from know_everything_ai.stores.base import SearchHit
from know_everything_ai.utils.llm_utils import LLMDriven, build_target


class FakeAgent:
    def __init__(
        self,
        *,
        model: str = "glm-5.3-flash",
        chunks: tuple[str, ...] = ("Отв", "ет."),
    ) -> None:
        self.model = model
        self.chunks = chunks
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def stream_answer(self, question: str, fragments: list[SearchHit]):
        for chunk in self.chunks:
            yield chunk


class ExplodingAgent(FakeAgent):
    async def stream_answer(self, question: str, fragments: list[SearchHit]):
        raise httpx.ConnectError("model down")
        yield  # pragma: no cover  - the raise keeps this an async generator


def hit(
    content: str,
    *,
    source: str | None = "https://files/x.md",
    title: str | None = "Раздел A",
    score: float = 0.9,
) -> SearchHit:
    return SearchHit(
        chunk_id=1,
        content=content,
        score=score,
        category="document",
        source=source,
        title=title,
    )


# ---------------------------------------------------------------- user message


def test_build_user_message_numbers_and_cites_fragments() -> None:
    message = build_answer_user_message(
        "как настроить?",
        [
            hit("Первый фрагмент.", title="Настройка", source="https://files/1.md"),
            hit("Второй фрагмент.", title=None, source=None),
        ],
    )

    assert "[1] (раздел: Настройка, источник: https://files/1.md)" in message
    assert "[2]" in message  # a fragment without title/source still gets an anchor
    assert "Первый фрагмент." in message
    assert "Второй фрагмент." in message
    assert message.endswith("Вопрос: как настроить?")


def test_build_user_message_clips_oversized_fragment() -> None:
    long_content = "ф" * 5000
    message = build_answer_user_message("вопрос", [hit(long_content)])
    assert "…" in message
    assert len(message) < 2500  # the fragment was cut to MAX_FRAGMENT_CHARS


def test_usage_delta_is_the_per_call_spend() -> None:
    before = {"calls": 3, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    after = {"calls": 4, "prompt_tokens": 22, "completion_tokens": 9, "total_tokens": 31}
    assert usage_delta(before, after) == {
        "calls": 1,
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }


# ------------------------------------------------------------------ event stream


@pytest.mark.asyncio
async def test_stream_events_full_turn() -> None:
    agent = FakeAgent()
    row = {"kb_external_id": "kb-1", "kb_type": "auto", "branch": "vector"}
    events = [
        event
        async for event in stream_answer_events(
            agent=agent,
            question="вопрос?",
            hits=[hit("Как настроить X."), hit("Шаг второй.")],
            row=row,
            floor=0.5,
        )
    ]

    assert [kind for kind, _ in events] == ["sources", "delta", "delta", "done"]

    sources = events[0][1]
    assert sources["kb_external_id"] == "kb-1"
    assert sources["branch"] == "vector"
    assert sources["floor"] == 0.5
    assert sources["model"] == "glm-5.3-flash"
    assert len(sources["hits"]) == 2
    first = sources["hits"][0]
    assert first["content"] == "Как настроить X."
    assert first["source"] == "https://files/x.md"
    assert first["title"] == "Раздел A"
    assert first["score"] == 0.9

    answer = "".join(payload["text"] for kind, payload in events if kind == "delta")
    assert answer == "Ответ."

    done = events[-1][1]
    assert done["hit_count"] == 2
    assert done["text_len"] == len("Ответ.")
    assert isinstance(done["took_ms"], int)
    assert done["usage"] == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.asyncio
async def test_stream_no_hits_never_touches_the_model() -> None:
    agent = FakeAgent()
    events = [
        event
        async for event in stream_answer_events(
            agent=agent,
            question="вопрос?",
            hits=[],
            row={"kb_external_id": "kb-1", "kb_type": "auto", "branch": "vector"},
            floor=-1.0,
        )
    ]

    assert [kind for kind, _ in events] == ["sources", "delta", "done"]
    assert events[1][1]["text"] == CANNED_NO_HITS
    assert events[2][1]["hit_count"] == 0
    assert events[2][1]["usage"] == {}


@pytest.mark.asyncio
async def test_stream_model_failure_becomes_an_error_event() -> None:
    agent = ExplodingAgent()
    events = [
        event
        async for event in stream_answer_events(
            agent=agent,
            question="вопрос?",
            hits=[hit("фрагмент")],
            row={"kb_external_id": "kb-1", "kb_type": "auto", "branch": "vector"},
            floor=-1.0,
        )
    ]

    assert [kind for kind, _ in events] == ["sources", "error"]
    assert "ConnectError" in events[1][1]["error"]


# --------------------------------------------------------------- complete_stream


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._items = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        if isinstance(item, BaseException):
            raise item
        return item


class FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.calls = 0

    async def create(self, **kwargs: Any) -> FakeStream:
        self.calls += 1
        return FakeStream(self.chunks)


class FakeChat:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(chunks))


def make_chunk(
    text: str | None = None, *, usage: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
        if text is not None
        else [],
        usage=usage,
    )


def usage_chunk(count: int) -> SimpleNamespace:
    return make_chunk(
        usage=SimpleNamespace(
            prompt_tokens=count,
            completion_tokens=count,
            total_tokens=count * 2,
        )
    )


def make_component(clients: dict[str, FakeChat]) -> LLMDriven:
    primary = build_target(
        api_url="http://primary", api_key="k", model="m1", timeout=1, label="primary"
    )
    fallback = build_target(
        api_url="http://fb", api_key="k", model="m2", timeout=1, label="fallback"
    )
    component = LLMDriven(
        api_url="http://primary",
        api_key="k",
        model="m1",
        system_prompt="",
        timeout=1,
    )
    component.targets = [primary, fallback]
    component._client_for = lambda target: clients[target.label]  # noqa: SLF001
    return component


@pytest.mark.asyncio
async def test_complete_stream_aggregates_deltas_and_usage() -> None:
    clients = {
        "primary": FakeChat(
            [make_chunk("Hel"), make_chunk("lo"), usage_chunk(3)]
        ),
        "fallback": FakeChat([]),
    }
    component = make_component(clients)

    parts = [
        part
        async for part in component.complete_stream(
            messages=[{"role": "user", "content": "x"}]
        )
    ]

    assert "".join(parts) == "Hello"
    assert component.usage["calls"] == 1
    assert component.usage["total_tokens"] == 6
    assert component._preferred_index == 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_complete_stream_fails_over_before_first_token() -> None:
    clients = {
        "primary": FakeChat([httpx.ConnectError("down")]),
        "fallback": FakeChat([make_chunk("fb"), usage_chunk(1)]),
    }
    component = make_component(clients)

    parts = [
        part
        async for part in component.complete_stream(
            messages=[{"role": "user", "content": "x"}]
        )
    ]

    assert "".join(parts) == "fb"
    assert component._preferred_index == 1  # noqa: SLF001
    assert clients["primary"].chat.completions.calls == 1
    assert clients["fallback"].chat.completions.calls == 1


@pytest.mark.asyncio
async def test_complete_stream_continues_after_mid_stream_failure() -> None:
    """A target that dies mid-response must not restart the question: the
    fallback picks up where the text stopped."""
    clients = {
        "primary": FakeChat(
            [make_chunk("Hel"), httpx.ConnectError("died"), usage_chunk(2)]
        ),
        "fallback": FakeChat([make_chunk("lo"), usage_chunk(1)]),
    }
    component = make_component(clients)

    parts = [
        part
        async for part in component.complete_stream(
            messages=[{"role": "user", "content": "x"}]
        )
    ]

    assert "".join(parts) == "Hello"
    assert component._preferred_index == 1  # noqa: SLF001
    # Usage rides on the final chunk of a stream, so the aborted primary's
    # usage chunk is never seen: only the fallback that completed is charged.
    assert component.usage["total_tokens"] == 2
