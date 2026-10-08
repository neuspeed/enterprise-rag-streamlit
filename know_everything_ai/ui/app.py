"""Sandbox dashboard for demonstrating and debugging the pipeline by hand.

This is a developer tool, not a production surface. It talks to
:class:`~know_everything_ai.pipeline.KnowledgePipeline` directly instead of
going through RabbitMQ, so an operator can watch one knowledge base go from an
upload to a reviewed payload without wiring up a broker.

The flow is deliberately split into three explicit steps — *measure*,
*process*, *push* — because the interesting behaviour of this pipeline is which
branch a document takes, and a single "run" button would hide exactly the thing
worth seeing.

Two implementation notes worth knowing before editing:

* Streamlit reruns the whole script on every interaction and has no event loop
  of its own. The pipeline owns an ``httpx`` pool bound to the loop that made
  its first request, so ``asyncio.run`` per click would leave the pool attached
  to a closed loop. A single background loop lives for the session instead.
* Measuring a PDF *is* a vision-model call, so the measured input is cached
  between the "measure" and "process" buttons and reused rather than reloaded.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import streamlit as st
from pydantic import AnyUrl

from know_everything_ai.agent.answer import AnswerAgent, stream_answer_events
from know_everything_ai.api.services import KbNotFoundError, KbNotReadyError, QueryService
from know_everything_ai.pipeline import (
    KnowledgePipeline,
    PreparedInput,
    PreviewResult,
)
from know_everything_ai.schemas import Payload
from know_everything_ai.settings import Settings

T = TypeVar("T")

ACCEPTED_SUFFIXES = {".pdf", ".docx", ".txt"}

#: How many characters of a generated element to show before truncating. A
#: knowledge canvas is one very long string and `st.code` on the full thing
#: makes the page unusable.
PREVIEW_CHARS = 2000


class SessionLoop:
    """One event loop, owned by the process, for the lifetime of the session.

    Every coroutine the UI runs is submitted here, so the pipeline's connection
    pool is created once and stays attached to a live loop. Closing the loop
    would break any later button press, so it is left running until exit.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._serve, name="ui-event-loop", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T], timeout: float) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Start a coroutine without blocking the caller.

        Used for chat streaming: the consumer runs on the UI thread and the
        coroutine pushes its events into a queue, so the two sides never wait on
        each other synchronously.
        """
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    @property
    def is_running(self) -> bool:
        return not self._loop.is_closed()


@st.cache_resource
def get_session_loop() -> SessionLoop:
    return SessionLoop()


@st.cache_resource
def get_pipeline() -> KnowledgePipeline:
    """The one pipeline instance behind every button on this page."""
    return KnowledgePipeline(build_sandbox_settings())


def build_sandbox_settings() -> Settings:
    """Settings for a local demo, with local files explicitly enabled.

    Uploads are handed to the loaders as paths on this filesystem, which the
    production defaults refuse. That refusal is correct there and is overridden
    here rather than globally, so a deployment never inherits it by accident.
    """
    return Settings(ALLOW_LOCAL_SOURCES=True)


@st.cache_resource
def upload_dir() -> Path:
    """Where uploads are written, kept next to the app rather than in DATA_DIR.

    ``DATA_DIR`` is a container path in the stand ``.env`` and resolves to the
    drive root when the UI runs on a developer machine. Relative to the working
    directory this lands in the same gitignored ``data/`` in the container
    (``/app/data/ui_uploads``) and on a workstation (``./data/ui_uploads``).
    The loaders copy uploads into their own scratch directory, so this choice
    does not affect where documents are read from.
    """
    target = Path.cwd() / "data" / "ui_uploads"
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_uploads(uploaded: list[Any]) -> list[str]:
    """Persist Streamlit uploads and return paths the loaders can open.

    Streamlit's ``UploadedFile`` handle is torn down on rerun, so the bytes are
    written to disk first. Names are sanitised: they come from a browser and
    would otherwise be able to escape the upload directory.
    """
    saved: list[str] = []
    for item in uploaded:
        name = Path(item.name).name
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        destination = upload_dir() / stem
        destination.write_bytes(item.getbuffer())
        saved.append(str(destination))
    return saved


def build_payload(
    *,
    kb_external_id: str,
    kb_type: Literal["auto", "context", "vector"],
    paths: list[str],
    text: str,
    external_url: str,
) -> Payload:
    """Assemble the same message shape the worker receives off the queue."""
    if paths:
        data: Any = paths if len(paths) > 1 else paths[0]
    elif text.strip():
        data = text.strip()
    else:
        raise ValueError("Загрузите файл или вставьте текст")
    return Payload(
        id=1,
        job_id=1,
        kb_type=kb_type,
        kb_external_id=kb_external_id,
        data=data,
        external_url=AnyUrl(external_url),
    )


def render_documents(prepared: PreparedInput) -> None:
    """What the corpus actually looks like after loading and cleaning."""
    documents = prepared.documents
    with st.expander(f"Исходные документы ({len(documents)})"):
        table = [
            {
                "source": (doc.source or "")[:60],
                "title": doc.title[:60],
                "category": doc.category.value
                if hasattr(doc.category, "value")
                else str(doc.category),
                "символы": len(doc.content),
            }
            for doc in documents
        ]
        st.dataframe(table, width="stretch")
        preview = documents[0].content[:PREVIEW_CHARS] if documents else ""
        if preview:
            st.markdown("**Первый документ, первые символы**")
            st.code(preview, language="markdown")


def render_measurement(prepared: PreparedInput) -> None:
    """Token count and branch — the decision the buyer never has to make."""
    columns = st.columns(4)
    columns[0].metric("Токены", f"{prepared.total_tokens:,}".replace(",", " "))
    columns[1].metric("Порог", f"{prepared.threshold:,}".replace(",", " "))
    columns[2].metric("Токенайзер", prepared.estimator)
    columns[3].metric("Документов", len(prepared.documents))

    usage = prepared.total_tokens / prepared.threshold if prepared.threshold else 0.0
    st.progress(min(usage, 1.0), text=f"Занято {usage:.1%} от порога")

    if prepared.branch == "context":
        st.success(
            "Ветка **context** — документ помещается в один длинный контекст. "
            "Будет собрано одно полотно знаний, без чанкинга."
        )
    else:
        st.warning(
            "Ветка **vector** — документ не помещается в один вызов. "
            "Будет разбит на чанки, отфильтрован и разобран на QA и глоссарий."
        )


def render_result(result: PreviewResult) -> None:
    """The generated payload and its structure, before anything is stored."""
    if not result.items:
        st.error("Ничего полезного не найдено — результат пуст.")
        return

    st.markdown(f"**Элементов: {len(result.items)}**")
    categories: dict[str, int] = {}
    for item in result.items:
        key = str(item.get("category", "—"))
        categories[key] = categories.get(key, 0) + 1
    st.write("Категории: " + ", ".join(f"`{k}` — {v}" for k, v in categories.items()))

    cost = result.cost
    columns = st.columns(4)
    columns[0].metric("Модельных вызовов", cost.get("model_calls", 0))
    columns[1].metric("Токены промпта", cost.get("prompt_tokens", 0))
    columns[2].metric("Токены ответа", cost.get("completion_tokens", 0))
    columns[3].metric("Всего токенов", cost.get("total_tokens", 0))

    with st.expander("Payload целиком (JSON)", expanded=True):
        st.json(result.items)
    with st.expander("Payload целиком (сырой текст ответа модели)"):
        st.code(
            json.dumps(result.items, ensure_ascii=False, indent=2), language="json"
        )


def _fmt_dt(value: Any) -> str:
    """Registry rows carry ``created_at`` as datetime.datetime, not a string."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return value.isoformat()[:19].replace("T", " ") if hasattr(value, "isoformat") else str(value)
    return value[:19].replace("T", " ")


def load_kb_rows(
    loop: SessionLoop, pipeline: KnowledgePipeline
) -> tuple[list[dict[str, Any]], int]:
    """All knowledge bases the read side can actually serve."""
    service = QueryService(
        build_sandbox_settings(), pipeline.stores, pipeline.stores.embedder
    )
    rows, total = loop.run(service.list_all(50, 0), timeout=60)
    return rows, total


def render_kb_panel(pipeline: KnowledgePipeline) -> None:
    """Shared management panel, shown on both tabs so nothing disappears.

    The list is read straight from the registry and vector store the chat
    answers from, and the selection feeds both the «Пайплайн» and «Чат» inputs.
    It is cached per session and reloaded after a write or on demand.
    """
    session = st.session_state
    st.subheader("Базы знаний")

    if session.pop("kb_dirty", False) or st.button(
        "Обновить список", key="kb_refresh"
    ):
        session.pop("kb_rows", None)

    if "kb_rows" not in session:
        with st.spinner("Читаю базы знаний из хранилища..."):
            try:
                rows, total = load_kb_rows(get_session_loop(), pipeline)
            except Exception as exc:  # DB may be unreachable from this container
                st.caption(f"Хранилище недоступно: {type(exc).__name__}: {exc}")
                session["kb_rows"] = []
                session["kb_total"] = 0
            else:
                session["kb_rows"] = rows
                session["kb_total"] = total

    after_save = session.pop("kb_after_save", None)
    if after_save:
        # Assigned before the widget below is created, which is the one order
        # Streamlit allows for retargeting a keyed widget in the same run.
        session["kb_picker"] = after_save
        session["chat_kb"] = after_save

    rows = session["kb_rows"]
    if not rows:
        st.caption(
            "Базы знаний ещё нет. Откройте вкладку «Пайплайн», обработайте "
            "документ и нажмите «Сохранить в базу знаний», чтобы появилась."
        )
        return

    by_id = {row["kb_external_id"]: row for row in rows}
    st.selectbox(
        "Рабочая база знаний",
        list(by_id),
        key="kb_picker",
        format_func=lambda kb: (
            f"{kb} — {by_id[kb]['branch']} · {by_id[kb].get('item_count', 0)} элементов"
        ),
    )
    with st.expander(f"Все базы знаний ({len(rows)})"):
        table = [
            {
                "kb_external_id": row["kb_external_id"],
                "ветка": row["branch"],
                "статус": row["status"],
                "элементов": row.get("item_count", 0),
                "чанков": row.get("chunk_count", 0),
                "токенов": row.get("total_tokens", 0),
                "создано": _fmt_dt(row.get("created_at")),
            }
            for row in rows
        ]
        st.dataframe(table, width="stretch", hide_index=True)


def sync_kb_from_picker() -> None:
    """Follow a new picker choice into mode inputs.

    Selecting a KB should prefill both the «Чат» and, if the «Пайплайн» field
    is empty/at default, also the «Пайплайн» (so the user can extend an
    existing KB). If the «Пайплайн» field was explicitly changed to a name that
    does not exist in the current picker list (a new KB), we must NOT overwrite
    it — otherwise the operator cannot type a new kb_external_id.
    """
    picker = st.session_state.get("kb_picker")
    if not picker:
        return
    if st.session_state.get("kb_picker_last_set") == picker:
        return

    rows = st.session_state.get("kb_rows") or []
    known = {row["kb_external_id"] for row in rows}

    if "kb_pipeline_last_autoset" not in st.session_state:
        st.session_state["kb_pipeline_last_autoset"] = st.session_state.get(
            "pipeline_kb"
        )

    st.session_state["chat_kb"] = picker

    pipe_cur = st.session_state.get("pipeline_kb") or ""
    last_autoset = st.session_state.get("kb_pipeline_last_autoset") or pipe_cur

    pipe_is_existing = pipe_cur in known
    if pipe_cur == last_autoset or pipe_is_existing:
        st.session_state["pipeline_kb"] = picker
        st.session_state["kb_pipeline_last_autoset"] = picker
    st.session_state["kb_picker_last_set"] = picker


#: The two ways a run can relate to a knowledge base: create one under the
#: given id (a re-run overwrites it), or extend one that already exists.
KB_MODE_NEW = "Новая база"
KB_MODE_APPEND = "Добавить к существующей"


def append_target_problem(row: dict[str, Any] | None) -> str | None:
    """Why files cannot be added to this knowledge base, or ``None`` if they can.

    Pure so the rule is testable without a browser: an append needs a row to
    exist and that row to be on a branch whose shape this run can extend.
    """
    if row is None:
        return (
            "База знаний не найдена. Добавлять файлы можно только к "
            "существующей: выберите её в панели «Базы знаний» или "
            "переключитесь на «Новая база»."
        )
    branch = str(row.get("branch") or "")
    if branch not in {"context", "vector"}:
        return (
            f"У базы неизвестная ветка «{branch}» — добавление в неё "
            "невозможно."
        )
    return None


def sidebar() -> tuple[str, Literal["auto", "context", "vector"], str, bool]:
    """The task definition: target KB, how it is related to, and the webhook.

    The fourth element is whether files are being added to an existing
    knowledge base. In that mode the branch selector is disabled: the branch
    belongs to the base being extended, not to this run, so it is pinned to
    whatever the target already is.
    """
    with st.sidebar:
        st.header("Задание")
        mode = st.radio(
            "Режим базы знаний",
            [KB_MODE_NEW, KB_MODE_APPEND],
            horizontal=True,
            key="pipeline_kb_mode",
            help=(
                "«Новая база» — создаёт базу с этим kb_external_id; повторный "
                "запуск перезаписывает её содержимое. «Добавить к существующей» "
                "— новые файлы дописываются к уже созданной базе, ничего из "
                "неё не удаляется."
            ),
        )
        append_mode = mode == KB_MODE_APPEND
        kb_external_id = st.text_input(
            "kb_external_id",
            key="pipeline_kb",
            help="Идентификатор базы знаний клиента. Document Store "
            "создаётся детерминированно как kb_<значение>. Выбор в "
            "панели «Базы знаний» подставится сюда.",
        )
        kb_type = cast(
            Literal["auto", "context", "vector"],
            st.selectbox(
                "Тип базы знаний",
                ["auto", "context", "vector"],
                disabled=append_mode,
                help=(
                    "Ветка берётся из существующей базы: добавление всегда "
                    "пишет в ту же ветку."
                    if append_mode
                    else "auto — ветка выбирается по размеру документа."
                ),
            ),
        )
        webhook = st.text_input(
            "Webhook получателя",
            value="http://webhook_sink:80/hook",
            help="Куда уйдёт ReturnPayload. В контейнере — имя сервиса.",
        )
        settings = build_sandbox_settings()
        st.header("Конфигурация")
        st.code(
            f"ENRICHER: {settings.ENRICHER_MODEL_NAME}\n"
            f"VLM (парсер PDF): {settings.VLM_MODEL_NAME}\n"
            f"CLASSIFIER: {settings.CLASSIFIER_MODEL_NAME}\n"
            f"FLOWISE_ENABLED: {settings.FLOWISE_ENABLED}\n"
            f"FLOWISE_API_URL: {settings.FLOWISE_API_URL or '—'}\n"
            f"Токенайзер: {settings.TOKENIZER_BACKEND}",
            language="text",
        )
    return kb_external_id, kb_type, webhook, append_mode


def chat_stream(
    loop: SessionLoop,
    **kwargs: Any,
) -> Any:
    """Events from the shared answer stream, pulled across the loop boundary.

    ``stream_answer_events`` runs on the session loop thread; its ``(kind,
    payload)`` pairs are pushed into a queue that the calling thread drains, so
    a long model stream never blocks the loop.
    """
    end = object()
    events: queue.Queue = queue.Queue()

    async def pump() -> None:
        try:
            async for kind, payload in stream_answer_events(**kwargs):
                events.put((kind, payload))
        except Exception as exc:  # surfaced in the UI, not swallowed
            events.put(("error", {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put((end, None))

    loop.submit(pump())
    while True:
        kind, payload = events.get()
        if kind is end:
            return
        yield kind, payload


def render_sources(payload: dict[str, Any]) -> None:
    """The fragments the model was allowed to read, before the answer starts."""
    hits = payload.get("hits", [])
    floor = payload.get("floor", 0.0)
    header = (
        f"Источники: {len(hits)} фрагментов (порог {floor}, модель {payload.get('model', '?')})"
    )
    if not hits:
        st.caption(header + " — ничего не найдено.")
        return
    with st.expander(header, expanded=bool(hits)):
        for hit in hits:
            title = hit.get("title") or hit.get("category") or "фрагмент"
            source = f" — {hit['source']}" if hit.get("source") else ""
            st.markdown(f"**{title}** · score {hit.get('score'):.3f}{source}")
            st.markdown((hit.get("content") or "")[:800])


def render_chat() -> None:
    """Chat tab: ask the buyer's knowledge base in plain language.

    The same :func:`stream_answer_events` feed the SSE endpoint uses, rendered
    natively. If ``ANSWER_*`` is not configured the model is skipped and the
    retrieval results are shown as a degraded read-only answer, mirroring the
    API's 503 without making the tab useless.
    """
    st.subheader("Чат с базой знаний")
    settings = build_sandbox_settings()
    agent = AnswerAgent(settings)
    pipeline = get_pipeline()

    if not agent.configured:
        st.caption(
            "Модель ответов не настроена (``ANSWER_API_URL``/``ANSWER_MODEL_NAME``) — "
            "будет показан только поиск по фрагментам. Задайте роль ``answer`` в `.env`, "
            "чтобы получить сгенерированный ответ."
        )

    kb_external_id = st.text_input(
        "kb_external_id",
        key="chat_kb",
        help="Идентификатор базы знаний, уже созданной. Выбирайте её в "
        "панели «Базы знаний», или введите вручную.",
    )
    question = st.text_area("Вопрос", height=100, placeholder="Например: как настроить двухэтапную проверку?")
    ask = st.button("Спросить", type="primary")
    if not ask:
        return
    if not kb_external_id.strip() or not question.strip():
        st.error("Укажите и базу знаний, и вопрос.")
        return

    st.caption(
        "Первый вопрос на сессию может занимать около минуты: скачивается "
        "встроенная в контейнер модель эмбеддингов."
    )

    loop = get_session_loop()
    service = QueryService(settings, pipeline.stores, pipeline.stores.embedder)
    limit = min(6, agent.max_fragments)
    with st.spinner("Ищу фрагменты по базе знаний..."):
        try:
            row, hits, took_ms, floor = loop.run(
                service.query(kb_external_id, question, limit, None), timeout=300
            )
        except KbNotFoundError:
            known = st.session_state.get("kb_rows") or []
            existing = ", ".join(r["kb_external_id"] for r in known) or "ни одной"
            st.error(
                f"База знаний «{kb_external_id}» не найдена. Создайте её на "
                f"вкладке «Пайплайн» или выберите из списка выше: {existing}"
            )
            return
        except KbNotReadyError as exc:
            st.error(f"База знаний ещё не готова: {exc}")
            return
        except TimeoutError:
            st.error(
                "Поиск по базе знаний не уложился в таймаут. Попробуйте ещё раз: "
                "первый вызов обычно самый долгий (загрузка эмбеддинг-модели)."
            )
            return
        except Exception as exc:  # surfaced in the UI, not swallowed
            st.error(f"Ошибка поиска: {type(exc).__name__}: {exc}")
            return

    st.caption(f"Поиск занял {took_ms} ms, порог {floor}, модель: {agent.model or '—'}.")

    if not agent.configured:
        # Degraded mode: no model role is wired, so show the retrieval results
        # as the answer instead of calling an endpoint that would 503.
        render_sources({
            "hits": [_hit_dict(hit) for hit in hits],
            "floor": floor,
            "model": agent.model,
            "kb_external_id": kb_external_id,
        })
        if hits:
            st.markdown("**Фрагменты (режим без модели):**")
            for hit in hits:
                st.markdown(hit.content)
        else:
            st.markdown("По этому вопросу ничего не найдено.")
        return

    events = chat_stream(
        loop,
        agent=agent,
        question=question,
        hits=hits,
        row=row,
        floor=floor,
    )
    try:
        kind0, sources = next(events)
    except StopIteration:
        return
    if kind0 == "error":
        st.error(sources.get("error", "Поток завершился с ошибкой."))
        return
    render_sources(sources)

    errors: list[str] = []

    def deltas() -> Any:
        for kind, payload in events:
            if kind == "delta":
                yield payload["text"]
            elif kind == "error":
                errors.append(payload.get("error", "unknown error"))

    try:
        st.chat_message("assistant").write_stream(deltas())
    except Exception as exc:  # surfaced in the UI, not swallowed
        st.error(f"Ответ не достримлен: {type(exc).__name__}: {exc}")
        return
    if errors:
        st.warning("Ответ прерван: " + errors[0])


def _hit_dict(hit: Any) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "content": hit.content,
        "score": hit.score,
        "category": hit.category,
        "source": hit.source,
        "title": hit.title,
        "metadata": dict(hit.metadata or {}),
    }


def main() -> None:
    st.set_page_config(page_title="Knowledge Pipeline — стенд", layout="wide")
    st.title("Knowledge Pipeline — стенд")
    mode = st.radio("Режим", ["Пайплайн", "Чат"], horizontal=True)
    pipeline = get_pipeline()
    # Widget defaults are seeded here once, then owned by the widgets' keys:
    # assigning their keys again later would trip Streamlit's "value set via
    # Session State" warning every rerun.
    st.session_state.setdefault("chat_kb", "client123_project_abc")
    st.session_state.setdefault("pipeline_kb", "client123_project_abc")
    render_kb_panel(pipeline)
    sync_kb_from_picker()
    if mode == "Чат":
        render_chat()
        return
    st.caption(
        "Загрузите документ, посмотрите, какая ветка выбрана, и проверьте "
        "payload до записи в базу."
    )

    kb_external_id, kb_type, webhook, append_mode = sidebar()

    # In append mode the run belongs to an existing base, so the target is
    # resolved from the registry (the source of truth, not the panel's cached
    # snapshot) and the branch is pinned to the one the base already has.
    effective_kb_type: Literal["auto", "context", "vector"] = kb_type
    if append_mode:
        try:
            target = get_session_loop().run(
                pipeline.stores.registry.get_by_external_id(kb_external_id),
                timeout=30,
            )
        except Exception as exc:
            st.error(f"Не удалось проверить базу знаний: {type(exc).__name__}: {exc}")
            return
        problem = append_target_problem(target)
        if target is None or problem is not None:
            # The helper reports a missing row too, so the message is always
            # set here; the `target is None` arm only exists for the type.
            st.error(f"«{kb_external_id}»: {problem}")
            return
        target_branch = str(target.get("branch") or "")
        effective_kb_type = cast(Literal["auto", "context", "vector"], target_branch)
        if target.get("status") not in (None, "ok"):
            st.warning(
                f"Статус базы «{kb_external_id}» — «{target.get('status')}»: "
                "она может быть неполной или не готовой к чтению."
            )
        tokens_text = f"{int(target.get('total_tokens') or 0):,}".replace(",", " ")
        st.info(
            f"Добавление в «{kb_external_id}»: ветка {target_branch}, уже в "
            f"базе {int(target.get('item_count') or 0)} элементов и "
            f"{tokens_text} токенов. Новые файлы будут дописаны, существующие "
            "не удалятся."
        )

    uploaded = st.file_uploader(
        "Документы",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Можно несколько файлов: они объединяются в одну базу знаний.",
    )
    pasted = st.text_area("Или вставьте текст напрямую", height=120)

    rejected = [
        item.name
        for item in (uploaded or [])
        if Path(item.name).suffix.lower() not in ACCEPTED_SUFFIXES
    ]
    if rejected:
        st.warning("Будут проигнорированы: " + ", ".join(rejected))

    if not (uploaded or pasted.strip()):
        st.info("Загрузите файл или вставьте текст, чтобы начать.")
        return

    session = st.session_state

    prepare_col, process_col = st.columns(2)
    do_prepare = prepare_col.button(
        "1. Разобрать и измерить", type="primary", use_container_width=True
    )
    do_process = process_col.button("2. Обработать", use_container_width=True)

    def current_payload() -> Payload:
        return build_payload(
            kb_external_id=kb_external_id,
            kb_type=effective_kb_type,
            paths=save_uploads(list(uploaded or [])),
            text=pasted,
            external_url=webhook,
        )

    if do_prepare:
        session.pop("result", None)
        with st.spinner("Загружаю документы и считаю токены..."):
            try:
                payload = current_payload()
                session["payload"] = payload
                session["prepared"] = get_session_loop().run(
                    pipeline.prepare(payload), timeout=600
                )
            except Exception as exc:  # surfaced in the UI, not swallowed
                st.error(f"Не удалось подготовить: {exc}")
                session.pop("prepared", None)
                return

    if "prepared" not in session:
        st.caption("Нажмите «1. Разобрать и измерить», чтобы увидеть токены.")
        return

    prepared: PreparedInput = session["prepared"]
    if prepared.branch != _requested_branch(effective_kb_type):
        st.caption(
            "Задание в очереди было другим: результат выше относится к прошлому "
            "нажатию. Нажмите «1. Разобрать и измерить», чтобы пересчитать."
        )

    render_measurement(prepared)
    render_documents(prepared)

    if do_process:
        with st.spinner("Готовлю знания..."):
            try:
                session["result"] = get_session_loop().run(
                    pipeline.run_preview(
                        session["payload"], prepared=prepared
                    ),
                    timeout=900,
                )
            except Exception as exc:
                st.error(f"Ошибка обработки: {exc}")
                session.pop("result", None)

    if "result" in session:
        result: PreviewResult = session["result"]
        st.divider()
        st.subheader("Результат")
        render_result(result)

        save_label = (
            "3. Добавить в базу знаний (для «Чата»)"
            if append_mode
            else "3. Сохранить в базу знаний (для «Чата»)"
        )
        if st.button(save_label, type="primary"):
            with st.spinner(
                "Дописываю новые чанки к базе..."
                if append_mode
                else "Разбираю результат на чанки и записываю в хранилище..."
            ):
                try:
                    kb_id = get_session_loop().run(
                        pipeline.store_into_own_store(
                            session["payload"],
                            result.items,
                            branch=result.branch,
                            total_tokens=result.total_tokens,
                            estimator=result.estimator,
                            threshold=prepared.threshold,
                            append=append_mode,
                        ),
                        timeout=300,
                    )
                except Exception as exc:
                    st.error(f"Не удалось сохранить: {exc}")
                else:
                    if append_mode:
                        st.success(
                            f"Добавлено к базе «{session['payload'].kb_external_id}» "
                            f"(id={kb_id}): теперь в ней на {len(result.items)} "
                            "элементов больше. Вкладка «Чат» подхватит изменения "
                            "сразу."
                        )
                    else:
                        st.success(
                            f"Сохранено (id={kb_id}). Переключитесь на «Чат» — "
                            f"база уже будет выбрана в панели «Базы знаний»."
                        )
                    st.session_state["kb_after_save"] = session[
                        "payload"
                    ].kb_external_id
                    st.session_state["kb_dirty"] = True

        if build_sandbox_settings().FLOWISE_ENABLED:
            push_label = (
                "4. Дописать в Flowise Document Store"
                if append_mode
                else "4. Записать в Flowise Document Store"
            )
            if st.button(push_label):
                with st.spinner(
                    "Нахожу store и дописываю документы..."
                    if append_mode
                    else "Создаю или нахожу store и записываю..."
                ):
                    try:
                        store = get_session_loop().run(
                            pipeline.push_to_store(
                                session["payload"],
                                result.items,
                                append=append_mode,
                            ),
                            timeout=300,
                        )
                        if append_mode:
                            st.success(
                                f"Дописано в `{store.name}` (id={store.id})."
                            )
                        else:
                            st.success(
                                f"Записано в `{store.name}` (id={store.id})."
                            )
                    except Exception as exc:
                        st.error(f"Не удалось записать: {exc}")
        else:
            # Offering a button that can only fail is worse than saying why.
            st.caption(
                "Запись в Flowise отключена: в `.env` стоит "
                "`FLOWISE_ENABLED=false`. Это не мешает «Сохранить в базу "
                "знаний» — она пишет во встроенное хранилище, из которого "
                "читает «Чат»."
            )


def _requested_branch(kb_type: str) -> str:
    return kb_type if kb_type in {"context", "vector"} else "auto"


if __name__ == "__main__":
    main()
