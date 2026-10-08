"""Ingestion pipeline and branching coordinator.

The buyer never picks a chunking strategy. Documents are measured, and the
measured size selects one of two branches:

* **context** — everything fits in one long-context call, so a single dense
  knowledge canvas is produced and returned as JSON;
* **vector** — too large for one call, so it is split, classified, structured
  and upserted into a vector store.

``kb_type`` may pin the branch; ``auto`` (the default) leaves the decision to
the measurement.

The pipeline holds one httpx pool, one Flowise connection pool and one set of
model clients for its whole life, and is reused across messages. Building these
per message leaked a connection pool per message and per file until the worker
ran out of file descriptors.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

import httpx
import structlog

from know_everything_ai.classifier.chunk_classifier import ChunkClassifier
from know_everything_ai.cleaners.text_cleaner import clean_text
from know_everything_ai.enricher.knowledge_canvas import KnowledgeCanvasEnricher
from know_everything_ai.flowise.client import Flowise
from know_everything_ai.flowise.models import (
    BaseConfig,
    DocumentStore,
    EmbeddingConfig,
    RecordManagerConfig,
    UpsertConfig,
    UpsertResult,
    VectorStoreConfig,
)
from know_everything_ai.flowise.transport import FlowiseTransport
from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.loaders.registry import create_loader
from know_everything_ai.schemas import (
    DocCategory,
    Payload,
    RawElement,
    ReturnPayload,
)
from know_everything_ai.settings import Settings
from know_everything_ai.stores import (
    Chunk,
    StoreContext,
    SuccessRecord,
    canvas_to_chunks,
)
from know_everything_ai.structurizers.faq_structurizer import FAQStructurizer
from know_everything_ai.structurizers.glossary_structurizer import GlossaryStructurizer
from know_everything_ai.utils.chunk_merger import ChunkMerger
from know_everything_ai.utils.llm_utils import LLMDriven, build_target
from know_everything_ai.utils.markdown_to_raw_elements import MarkdownToRawElements
from know_everything_ai.utils.source_client import SourceFetchError, guess_extension
from know_everything_ai.utils.token_counter import TokenEstimator
from know_everything_ai.utils.webhook_utils import WebhookSender

log = structlog.get_logger("knowledge_pipeline")

#: Branch names, matching ``ReturnPayload.kb_type``.
BranchType = Literal["vector", "context"]


class ResolvedStore(NamedTuple):
    """A Flowise document store with the fields the upsert path needs.

    ``DocumentStore`` declares both as optional because that is what the API
    returns; they are required here and checked on the way out, so the upsert
    never has to re-narrow them.
    """

    id: str
    name: str
    store: DocumentStore

@dataclass(frozen=True)
class PreparedInput:
    """A loaded, cleaned and measured knowledge base, before any LLM work.

    Exposed separately from :meth:`KnowledgePipeline.run` because loading a PDF
    *is* the vision-model call. An operator previewing a knowledge base needs the
    token count and the branch decision first, and re-loading to get them would
    pay for the same pages twice.
    """

    documents: list[RawElement]
    total_tokens: int
    branch: BranchType
    estimator: str
    threshold: int


@dataclass(frozen=True)
class PreviewResult:
    """Processed items plus the accounting, with nothing pushed anywhere."""

    items: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    branch: BranchType = "context"
    estimator: str = ""
    cost: dict[str, Any] = field(default_factory=dict)


NOTHING_USEFUL = "All data classified as not useful"

#: Source URL extension -> loader key. Anything unmapped is handed to the plain
#: text loader, which decodes bytes and cleans them rather than failing.
_EXTENSION_TO_LOADER = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
}


def _guess_source_type(url: str) -> str:
    """Pick a loader key for a bare source URL."""
    return _EXTENSION_TO_LOADER.get(guess_extension(url), "txt")


def _to_chunks(items: list[dict[str, Any]]) -> list[Chunk]:
    """Map processed vector-branch output onto the store's fragment shape.

    Kept separate from the context branch's sectioning on purpose: these items
    already went through the splitter, the classifier and the structurizers, so
    re-cutting them would undo that work.
    """
    chunks: list[Chunk] = []
    for item in items:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        category = item.get("category")
        if isinstance(category, DocCategory):
            category = category.value
        chunks.append(
            Chunk(
                content=content,
                category=category,
                source=item.get("source"),
                title=item.get("title"),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return chunks


class QuotaExceeded(ValueError):
    """The request exceeds a limit configured for a self-hosted deployment."""


class KnowledgePipeline:
    KB_NAME_PREFIX = "kb_"

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        stores: StoreContext | None = None,
    ) -> None:
        self.settings = settings
        # One pool for every model call. Owned here when the caller injects
        # nothing, released in aclose().
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.LLM_TIMEOUT)
        )
        self._owns_http_client = http_client is None
        # Built lazily on first use: a preview that never persists must not
        # require a reachable database or 220 MB of embedding weights.
        self.stores = stores or StoreContext(settings)

        self.enricher = KnowledgeCanvasEnricher(
            api_url=settings.ENRICHER_API_URL,
            api_key=settings.ENRICHER_API_KEY,
            model=settings.ENRICHER_MODEL_NAME,
            system_prompt=settings.ENRICHER_PROMPT,
            timeout=settings.ENRICHER_TIMEOUT,
            max_tokens=settings.ENRICHER_MAX_TOKENS,
            http_client=self._http_client,
            usage_name="enricher",
            fallback=build_target(
                api_url=settings.ENRICHER_FALLBACK_API_URL,
                api_key=settings.ENRICHER_FALLBACK_API_KEY,
                model=settings.ENRICHER_FALLBACK_MODEL_NAME,
                timeout=settings.ENRICHER_TIMEOUT,
                max_tokens=settings.ENRICHER_MAX_TOKENS,
                label="fallback",
            ),
        )
        self.classifier = ChunkClassifier(
            api_url=settings.CLASSIFIER_API_URL,
            api_key=settings.CLASSIFIER_API_KEY,
            model=settings.CLASSIFIER_MODEL_NAME,
            system_prompt=settings.CLASSIFIER_PROMPT_TEMPLATE,
            category_descriptions=settings.CLASSIFIER_CATEGORY_DESCRIPTIONS,
            trash_size=settings.CLASSIFIER_TRASH_SIZE,
            http_client=self._http_client,
            usage_name="classifier",
            fallback=build_target(
                api_url=settings.CLASSIFIER_FALLBACK_API_URL,
                api_key=settings.CLASSIFIER_FALLBACK_API_KEY,
                model=settings.CLASSIFIER_FALLBACK_MODEL_NAME,
                # The classifier passes no explicit timeout, so it runs on the
                # shared LLM_TIMEOUT; the fallback must match the primary.
                timeout=settings.LLM_TIMEOUT,
                label="fallback",
            ),
        )
        self.faq_struct = FAQStructurizer(settings)
        self.glossary_struct = GlossaryStructurizer(settings)
        self.splitter = MarkdownToRawElements()
        self.token_estimator = TokenEstimator(settings)
        # Shares the pipeline's estimator so merging uses the configured
        # tokenizer backend rather than a separate default.
        self.chunk_merger = ChunkMerger(estimator=self.token_estimator)
        self.webhook_sender = WebhookSender(settings)
        self.max_enrich_tokens = settings.MAX_CKB_TOKENS

        self.flowise: Flowise | None = None
        if settings.FLOWISE_ENABLED:
            self.flowise = Flowise(
                settings.FLOWISE_API_URL,
                settings.FLOWISE_API_KEY,
                transport=FlowiseTransport(
                    settings.FLOWISE_API_URL,
                    settings.FLOWISE_API_KEY,
                    timeout=settings.FLOWISE_TIMEOUT,
                    upsert_timeout=settings.FLOWISE_UPSERT_TIMEOUT,
                ),
            )
        self.embedding = EmbeddingConfig.huggingface_inference(
            model=settings.EMBEDDING_MODEL,
            credential=settings.EMBEDDING_CREDENTIAL,
            api_url=settings.EMBEDDING_API_URL,
        )

        # Loaders are cached per source type: each one owns a parser, and the
        # PDF parser owns a concurrency semaphore. Building one per file reset
        # that semaphore every time, so MAX_CONCURRENT_PDFS limited nothing.
        self._loaders: dict[str, BaseLoader] = {}

    # ------------------------------------------------------------------ lifecycle

    async def aclose(self) -> None:
        if self.flowise is not None:
            await self.flowise.aclose()
        await self.stores.aclose()
        if self._owns_http_client:
            await self._http_client.aclose()

    def _loader_for(self, source_type: str) -> BaseLoader:
        key = source_type.lower().lstrip(".")
        if key not in self._loaders:
            self._loaders[key] = create_loader(key, self.settings)
        return self._loaders[key]

    # -------------------------------------------------------------------- context

    def _reset_usage(self) -> None:
        """Zero the per-message counters and re-prioritise the primary provider.

        Both happen per message on purpose: usage is the buyer's bill for this
        job, and the target preference is per-job too, so a provider that was
        rate-limited on the last message is tried first again on this one and a
        provider that has recovered is picked up without a restart.
        """
        self.enricher.reset_usage()
        self.classifier.reset_usage()
        for component in self._llm_components():
            component.reset_usage()
            component.reset_targets()

    def _llm_components(self) -> tuple[LLMDriven, ...]:
        """Every model-backed component reachable from this pipeline.

        Loaders are cached across messages, so a VLM parser that ran for the
        previous message still holds that message's usage. Resetting and reading
        through one accessor keeps the ledger per-message instead of cumulative.
        """
        return (
            self.enricher,
            self.classifier,
            *(c for loader in self._loaders.values() for c in loader.llm_components()),
        )

    def _collect_cost(self, total_tokens: int) -> dict[str, Any]:
        """Aggregate token spend across every component for this message.

        ``usage_ledger`` in P1 is billed from this, so it has to be complete
        rather than best-effort: the inherited code wrote ``cost={}`` at all
        nine call sites and logged ``response.usage`` straight into the void.
        """
        by_component: dict[str, dict[str, int]] = {}
        for component in self._llm_components():
            usage = component.usage
            name = getattr(component, "usage_name", None) or type(component).__name__
            aggregate = by_component.setdefault(
                name,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            for metric in aggregate:
                aggregate[metric] += usage.get(metric, 0)

        totals = {
            metric: sum(usage[metric] for usage in by_component.values())
            for metric in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        return {
            "total_tokens": total_tokens,
            "model_calls": totals["calls"],
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "model_total_tokens": totals["total_tokens"],
            "by_component": by_component,
        }

    # ------------------------------------------------------------------------ run

    async def prepare(self, payload: Payload) -> PreparedInput:
        """Load, clean, measure and choose a branch — without calling any LLM.

        Split out of :meth:`run` so an operator can be shown the token count and
        the branch before spending a vision-model pass over the pages.
        """
        if not payload.kb_external_id:
            raise ValueError("kb_external_id is required")

        log.debug("documents_loading_start")
        raw_documents = await self._load_all_documents(payload)
        if not raw_documents:
            raise SourceFetchError("no documents could be loaded from the payload")

        # Cleaning and tokenizing walk every character of the corpus; keeping
        # them off the event loop matters once a knowledge base is large.
        merged_chunks = await asyncio.to_thread(self.chunk_merger.merge, raw_documents)
        if not merged_chunks:
            raise ValueError("documents merged into nothing")

        log.debug("documents_cleaning_start")
        for doc in merged_chunks:
            doc.content = clean_text(doc.content)
        log.debug("documents_cleaning_done")

        log.debug("token_counting_start")
        total_tokens = await asyncio.to_thread(self._count_tokens, merged_chunks)
        estimator = self.token_estimator.active_backend(
            self.settings.ENRICHER_MODEL_NAME
        )
        log.info(
            "token_estimate",
            total_tokens=total_tokens,
            estimator=estimator,
            threshold=self.max_enrich_tokens,
            backend=settings_backend(self.settings),
        )

        branch = self._select_branch(payload.kb_type, total_tokens)
        log.info(
            "branch_selected",
            kb_type=payload.kb_type,
            branch=branch,
            total_tokens=total_tokens,
            threshold=self.max_enrich_tokens,
        )

        return PreparedInput(
            documents=merged_chunks,
            total_tokens=total_tokens,
            branch=branch,
            estimator=estimator,
            threshold=self.max_enrich_tokens,
        )

    async def run_preview(
        self,
        payload: Payload,
        *,
        prepared: PreparedInput | None = None,
    ) -> PreviewResult:
        """Process a knowledge base and hand back the items instead of pushing.

        Nothing is written to Flowise and no webhook is sent, so the result can
        be inspected on screen first. Pushing stays an explicit second step.

        ``prepared`` lets a caller that already measured the corpus reuse that
        work. Re-preparing would reload the sources, and for a PDF that means a
        second vision-model pass over every page.
        """
        self._reset_usage()
        if prepared is None:
            prepared = await self.prepare(payload)
        items = await self._process(prepared)
        return PreviewResult(
            items=items,
            total_tokens=prepared.total_tokens,
            branch=prepared.branch,
            estimator=prepared.estimator,
            cost=self._collect_cost(prepared.total_tokens),
        )

    async def push_to_store(
        self,
        payload: Payload,
        items: list[dict[str, Any]],
        *,
        append: bool = False,
    ) -> DocumentStore:
        """Upsert already-processed items into this KB's deterministic store.

        Separate from :meth:`run` so an operator can inspect the output before
        anything is written. Routing is unchanged: the store name is always
        ``kb_<kb_external_id>`` and is created on first use.

        ``append=True`` extends an existing store instead of re-syncing it: the
        record manager runs without cleanup, so this batch only adds documents
        and the ones already indexed stay untouched.
        """
        if not items:
            raise ValueError("Nothing to push: the knowledge base is empty")
        resolved = await self._get_or_create_document_store(payload.kb_external_id)
        await self._flowise_upsert_documents(
            items,
            ds=resolved,
            cleanup_mode="none" if append else "full",
        )
        return resolved.store

    async def store_into_own_store(
        self,
        payload: Payload,
        items: list[dict[str, Any]],
        *,
        branch: str,
        total_tokens: int,
        estimator: str,
        threshold: int,
        append: bool = False,
    ) -> int:
        """Persist preview items into this KB's built-in store, no external push.

        :meth:`run_preview` is side-effect free so an operator can inspect the
        output first; this is the write half that makes the result answerable by
        the read side (QueryService/chat). Unlike :meth:`run` it sends no
        webhook and touches Flowise only if the vector branch is configured so
        end to end — the two stores stay in sync the same way the worker runs
        them.

        ``append=True`` adds this run's fragments to an existing knowledge base
        instead of replacing them: nothing stored earlier is deleted or
        re-embedded, the registry counters grow by this run's contribution and
        a context KB keeps its previous canvas entries. The target must exist
        and already be on the same branch — a mismatch is an error rather than
        a silent branch switch, because the caller chose the mode and the base
        was produced by a different one.
        """
        if not items:
            raise ValueError("Nothing to store: the knowledge base is empty")

        existing: dict[str, Any] | None = None
        if append:
            existing = await self.stores.registry.get_by_external_id(
                payload.kb_external_id
            )
            if existing is None:
                raise ValueError(
                    f"Knowledge base {payload.kb_external_id!r} does not exist yet; "
                    f"create it in the \"new\" mode before adding files to it"
                )
            if existing.get("branch") != branch:
                raise ValueError(
                    f"Knowledge base {payload.kb_external_id!r} is "
                    f"{existing.get('branch')!r}, but this run produced {branch!r} "
                    f"output; appending would mix two incompatible shapes"
                )

        if branch == "context":
            fragments = canvas_to_chunks(
                items,
                max_tokens=self.settings.CHUNK_SIZE_TOKENS,
                overlap_tokens=self.settings.CHUNK_OVERLAP_TOKENS,
                estimator=self.token_estimator,
            )
            canvas_tokens = sum(
                self.token_estimator.estimate(str(item.get("content", "")))
                for item in items
            )
            canvas = items
        else:
            fragments = _to_chunks(items)
            canvas_tokens = None
            canvas = None

        document_store_id: str | None = None
        if self.flowise is not None:
            store = await self._get_or_create_document_store(payload.kb_external_id)
            await self._flowise_upsert_documents(
                items,
                ds=store,
                cleanup_mode="none" if append else "full",
            )
            document_store_id = store.id or None

        if append and existing is not None:
            kb_id = int(existing["id"])
            await self.stores.vector_store.append_chunks(kb_id, fragments)
        else:
            kb_id = await self.stores.registry.ensure(
                payload.kb_external_id, payload.kb_type, branch
            )
            await self.stores.vector_store.replace_chunks(kb_id, fragments)

        # The row describes the whole knowledge base, not just this run: after
        # an append its size is what was there plus what was added, and a
        # context canvas is the concatenation of both generations.
        record_total_tokens = total_tokens
        record_item_count = len(items)
        record_canvas = canvas
        record_canvas_tokens = canvas_tokens
        if append and existing is not None:
            record_total_tokens = int(existing.get("total_tokens") or 0) + total_tokens
            record_item_count = int(existing.get("item_count") or 0) + len(items)
            if canvas is not None:
                previous_canvas = existing.get("canvas")
                record_canvas = (
                    list(previous_canvas) + canvas
                    if isinstance(previous_canvas, list)
                    else canvas
                )
                record_canvas_tokens = int(existing.get("canvas_tokens") or 0) + int(
                    canvas_tokens or 0
                )

        await self.stores.registry.record_success(
            SuccessRecord(
                kb_external_id=payload.kb_external_id,
                kb_type=payload.kb_type,
                branch=branch,
                total_tokens=record_total_tokens,
                item_count=record_item_count,
                estimator=estimator,
                threshold=threshold,
                payload_id=payload.id,
                job_id=payload.job_id,
                document_store_id=document_store_id,
                canvas_tokens=record_canvas_tokens,
                cost=self._collect_cost(total_tokens),
                canvas=record_canvas,
            )
        )
        return kb_id

    async def _process(self, prepared: PreparedInput) -> list[dict[str, Any]]:
        """Run the branch chosen in :meth:`prepare`, with no side effects."""
        if prepared.branch == "context":
            return await self._process_as_small(
                prepared.documents, prepared.total_tokens
            )
        return await self._process_as_large(prepared.documents)

    async def run(self, payload: Payload) -> int:
        """Process one message. Returns the number of stored items."""
        self._reset_usage()
        prepared = await self.prepare(payload)
        if prepared.branch == "context":
            return await self._run_context_branch(payload, prepared)
        return await self._run_vector_branch(payload, prepared)

    def _count_tokens(self, documents: list[RawElement]) -> int:
        return sum(self.token_estimator.estimate(doc.content) for doc in documents)

    def _select_branch(self, kb_type: str, total_tokens: int) -> BranchType:
        """Decide which branch to run.

        An explicit ``context`` request that does not fit is rejected rather
        than silently downgraded: the caller asked for one canvas and would
        otherwise receive a vector store it cannot read as one.
        """
        if kb_type == "context":
            if total_tokens > self.max_enrich_tokens:
                raise QuotaExceeded(
                    f"Cannot create context KB: total tokens {total_tokens} "
                    f"exceeds limit {self.max_enrich_tokens}"
                )
            return "context"
        if kb_type == "vector":
            return "vector"
        return "context" if total_tokens <= self.max_enrich_tokens else "vector"

    async def _run_context_branch(
        self, payload: Payload, prepared: PreparedInput
    ) -> int:
        documents, total_tokens = prepared.documents, prepared.total_tokens
        log.debug("processing_as_ckb_start")
        output_items = await self._process_as_small(documents, total_tokens)
        if not output_items:
            log.warning("pipeline_nothing_useful", branch="context")
            await self._notify_error(payload, "context", NOTHING_USEFUL)
            return 0

        # Identity first, "ok" second. A crash between them leaves the row
        # marked pending instead of reporting a canvas that was never stored,
        # and a re-run of an already successful KB keeps its old status until
        # this one succeeds, because `ensure` writes identity columns only.
        kb_id = await self.stores.registry.ensure(
            payload.kb_external_id, payload.kb_type, "context"
        )

        # The canvas is a single blob in the webhook, so without this it could
        # only be answered by re-running the job. Sections come from its JSON
        # structure, which keeps an entry and its definition in one fragment.
        fragments = canvas_to_chunks(
            output_items,
            max_tokens=self.settings.CHUNK_SIZE_TOKENS,
            overlap_tokens=self.settings.CHUNK_OVERLAP_TOKENS,
            estimator=self.token_estimator,
        )
        stored = await self.stores.vector_store.replace_chunks(kb_id, fragments)
        canvas_tokens = sum(
            self.token_estimator.estimate(str(item.get("content", "")))
            for item in output_items
        )

        await self.stores.registry.record_success(
            SuccessRecord(
                kb_external_id=payload.kb_external_id,
                kb_type=payload.kb_type,
                branch="context",
                total_tokens=total_tokens,
                item_count=len(output_items),
                estimator=prepared.estimator,
                threshold=prepared.threshold,
                payload_id=payload.id,
                job_id=payload.job_id,
                canvas_tokens=canvas_tokens,
                cost=self._collect_cost(total_tokens),
                canvas=output_items,
            )
        )

        await self._notify_success(
            payload,
            kb_type="context",
            # Unchanged: the context contract returns the canvas itself, and
            # the registry keeps a second copy for the read side.
            text_out=json.dumps(output_items, ensure_ascii=False),
            total_tokens=total_tokens,
        )
        log.debug(
            "processing_as_ckb_done", items=len(output_items), fragments=stored
        )
        return len(output_items)

    async def _run_vector_branch(
        self, payload: Payload, prepared: PreparedInput
    ) -> int:
        documents, total_tokens = prepared.documents, prepared.total_tokens
        log.debug("processing_as_vkb_start")
        output_items = await self._process_as_large(documents)
        if not output_items:
            log.warning("pipeline_nothing_useful", branch="vector")
            await self._notify_error(payload, "vector", NOTHING_USEFUL)
            return 0

        # The store is resolved only now: a context KB never touches Flowise,
        # so it no longer creates an empty kb_<id> store as a side effect.
        # Flowise is optional — when it is off the built-in store holds the
        # result and there is nothing to fail.
        document_store_id: str | None = None
        if self.flowise is not None:
            store = await self._get_or_create_document_store(payload.kb_external_id)
            await self._flowise_upsert_documents(output_items, ds=store)
            document_store_id = store.id or None

        kb_id = await self.stores.registry.ensure(
            payload.kb_external_id, payload.kb_type, "vector"
        )
        stored = await self.stores.vector_store.replace_chunks(
            kb_id, _to_chunks(output_items)
        )

        await self.stores.registry.record_success(
            SuccessRecord(
                kb_external_id=payload.kb_external_id,
                kb_type=payload.kb_type,
                branch="vector",
                total_tokens=total_tokens,
                item_count=len(output_items),
                estimator=prepared.estimator,
                threshold=prepared.threshold,
                payload_id=payload.id,
                job_id=payload.job_id,
                document_store_id=document_store_id,
                cost=self._collect_cost(total_tokens),
            )
        )

        await self._notify_success(
            payload,
            kb_type="vector",
            # The registry row, not the Flowise store id: the id used to be
            # the only handle on a knowledge base and it pointed at a service
            # that may be switched off. Flowise's own id travels separately.
            text_out=str(kb_id),
            total_tokens=total_tokens,
            document_store_id=document_store_id,
        )
        log.debug(
            "processing_as_vkb_done", items=len(output_items), fragments=stored
        )
        return len(output_items)

    # --------------------------------------------------------------------- notify

    async def _notify_success(
        self,
        payload: Payload,
        *,
        kb_type: BranchType,
        text_out: str,
        total_tokens: int,
        document_store_id: str | None = None,
    ) -> None:
        await self.webhook_sender.send_payload(
            str(payload.external_url),
            ReturnPayload(
                id=payload.id,
                job_id=payload.job_id,
                text_out=text_out,
                cost=self._collect_cost(total_tokens),
                kb_type=kb_type,
                status="success",
                document_store_id=document_store_id,
            ),
        )

    async def _notify_error(self, payload: Payload, kb_type: BranchType, reason: str) -> None:
        await self.webhook_sender.send_payload(
            str(payload.external_url),
            ReturnPayload(
                id=payload.id,
                job_id=payload.job_id,
                text_out="",
                cost=self._collect_cost(0),
                kb_type=kb_type,
                status="error",
                reason=reason,
            ),
        )

    # ------------------------------------------------------------------ documents

    async def _load_all_documents(self, payload: Payload) -> list[RawElement]:
        data = self._parse_payload_data(payload)
        all_docs: list[RawElement] = []

        files: list[tuple[str, str]] = [
            (source_type, url)
            for source_type, urls in (data.get("files") or {}).items()
            for url in urls
        ]
        files.extend(("html", url) for url in data.get("html") or [])

        # Count before scheduling anything: a coroutine created for a rejected
        # batch is work that starts and is then thrown away.
        file_budget = self.settings.MAX_FILES_PER_KB
        if len(files) > file_budget:
            raise QuotaExceeded(
                f"Too many files in one knowledge base: "
                f"{len(files)} given, limit is {file_budget}"
            )

        document_budget = self.settings.MAX_DOCUMENTS_PER_KB
        text = data.get("text")
        if text and text.strip():
            document_count = len(text)
        else:
            document_count = 0
        if document_count > document_budget:
            raise QuotaExceeded(
                f"Too many pasted documents in one knowledge base: "
                f"{document_count} given, limit is {document_budget}"
            )

        tasks: list = [
            self._loader_for(source_type).transform(url)
            for source_type, url in files
        ]

        if data.get("text"):
            all_docs.append(
                RawElement(
                    content=data["text"],
                    source="inline_text",
                    category=DocCategory.DOCUMENT,
                    title="",
                    metadata={},
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, BaseException):
                # One unreadable file must not sink the whole knowledge base.
                log.error("load_document_failed", error=str(result))
                failures.append(result)
                continue
            all_docs.extend(result)

        if failures and not all_docs:
            # Every source failed, so there is nothing to classify. Returning an
            # empty list here would be reported to the buyer as "nothing useful"
            # and acked as a finished job: a model provider answering 429, or one
            # dropped TCP connection, would quietly produce an empty knowledge
            # base instead of being retried. Re-raising keeps the dispatcher's
            # classification honest — a fetch error is permanent and goes to the
            # DLQ, a transient one is retried.
            raise failures[0]

        if len(all_docs) > self.settings.MAX_DOCUMENTS_PER_KB:
            raise QuotaExceeded(
                f"Too many document fragments: {len(all_docs)} exceeds "
                f"limit {self.settings.MAX_DOCUMENTS_PER_KB}"
            )
        return all_docs

    @staticmethod
    def _parse_payload_data(payload: Payload) -> dict:
        data = payload.data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # Payload documents "URL-строка" as an accepted shape, so a bare
                # URL has to be loaded as a document. Treating it as text sent a
                # 30-character link to the enricher instead of the file behind
                # it, which looked like a model failure rather than a parse one.
                stripped = data.strip()
                if stripped.startswith(("http://", "https://")):
                    return {
                        "files": {_guess_source_type(stripped): [stripped]},
                    }
                # A path to a file that exists is a source, not prose. The loaders
                # enforce ALLOW_LOCAL_SOURCES themselves and name the setting when
                # it is off; routing it here is what turns "the loader got a
                # 22-character file path as its document" into an explicit error.
                local = Path(stripped.replace("\\/", "/")).expanduser()
                if local.is_file():
                    return {
                        "files": {_guess_source_type(stripped): [stripped]},
                    }
                # A bare string is the document itself, not an envelope.
                return {"text": data}
        if isinstance(data, list):
            return {"files": {"txt": data}} if data else {}
        if not isinstance(data, dict):
            raise SourceFetchError(f"Unsupported payload data type: {type(data).__name__}")
        return data

    # --------------------------------------------------------------------- branches

    async def _process_as_small(
        self, raw_docs: list[RawElement], total_tokens: int = 0
    ) -> list[dict]:
        """Merge every document into one context canvas. No chunking."""
        combined_text = "\n\n".join(
            f"# Источник: {doc.source}\n{doc.content}" for doc in raw_docs
        )
        enriched = await self.enricher.create_knowledge_canvas(combined_text)

        return [
            {
                "content": enriched,
                "category": DocCategory.KNOWLEDGE_CANVAS.value,
                "source": "merged_documents",
                "created_at": None,
                "updated_at": None,
                "url": None,
                "metadata": {
                    "original_documents": list(
                        dict.fromkeys(doc.source for doc in raw_docs)
                    ),
                    "total_original_tokens": total_tokens,
                },
            }
        ]

    async def _process_as_large(self, raw_docs: list[RawElement]) -> list[dict]:
        """Split, classify and structure every document."""
        all_items: list[dict] = []
        for doc in raw_docs:
            structural_chunks = await asyncio.to_thread(
                self.splitter.convert,
                doc.content,
                source=doc.source or "",
                title=doc.title,
                url=doc.url,
                metadata=doc.metadata,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )

            for chunk in structural_chunks:
                classified = await self.classifier.classify(chunk)
                log.debug(
                    "classifier_result",
                    source=chunk.source,
                    category=classified.category.value,
                    is_useful=classified.is_useful,
                    reason=classified.reason,
                )
                if not classified.is_useful:
                    continue

                if classified.category == DocCategory.FAQ:
                    items = await self.faq_struct.extract(chunk)
                elif classified.category == DocCategory.GLOSSARY:
                    items = await self.glossary_struct.extract(chunk)
                else:
                    items = [
                        {
                            "content": chunk.content,
                            "category": classified.category.value,
                            "metadata": chunk.metadata,
                            "source": chunk.source,
                            "title": chunk.title,
                        }
                    ]
                all_items.extend(items)

        return all_items

    # ---------------------------------------------------------------------- flowise

    async def _get_or_create_document_store(self, kb_external_id: str) -> ResolvedStore:
        """Resolve the store for this KB, creating it on first use.

        The name is deterministic (``kb_<kb_external_id>``) so a retry lands in
        the same store instead of orphaning a half-populated one.
        """
        if self.flowise is None:
            raise RuntimeError(
                "Flowise is disabled; set FLOWISE_ENABLED=true and configure "
                "FLOWISE_API_URL to build a vector knowledge base"
            )

        ds_name = f"{self.KB_NAME_PREFIX}{kb_external_id}"
        existing = await self.flowise.document_store.find_document_store_by_name(ds_name)
        if existing:
            store = existing
        else:
            store = await self.flowise.document_store.create_document_store(
                store=DocumentStore(name=ds_name)
            )

        # Both fields are optional in the response model and both are needed
        # here: the name builds the vector table, the id is the upsert path. A
        # store returned without them would address /document-store/upsert/None
        # and a table called "none_vec_table".
        if not store.id or not store.name:
            raise RuntimeError(
                f"Flowise returned a document store without an id or name for "
                f"{ds_name!r}; the upsert would target the wrong location"
            )
        return ResolvedStore(id=store.id, name=store.name, store=store)

    async def _flowise_upsert_documents(
        self,
        items: list[dict],
        ds: ResolvedStore,
        *,
        cleanup_mode: str = "full",
    ) -> UpsertResult:
        if self.flowise is None:
            raise RuntimeError("Flowise is disabled; cannot upsert documents")

        table_name = f"{ds.name.lower().replace('-', '_')}_vec_table"
        vector_store = VectorStoreConfig.postgres(
            host=self.settings.VECTOR_STORE_HOST,
            port=self.settings.VECTOR_STORE_PORT,
            database=self.settings.VECTOR_STORE_DATABASE,
            table=table_name,
            topK=self.settings.VECTOR_STORE_TOPK,
            credential=self.settings.VECTOR_STORE_CREDENTIAL,
        )
        record_manager = RecordManagerConfig.postgres(
            host=self.settings.RECORD_MANAGER_HOST,
            port=self.settings.RECORD_MANAGER_PORT,
            database=self.settings.RECORD_MANAGER_DATABASE,
            table=f"{table_name}_upsertion_records",
            credential=self.settings.VECTOR_STORE_CREDENTIAL,
            # "full" keeps the default the worker has always run with: the
            # record manager may clean up whatever falls outside this batch.
            # An append must never do that — the documents outside the batch
            # are exactly the ones being kept — so it switches to "none",
            # where the manager only records what it indexed.
            cleanup_mode=cleanup_mode,
        )
        loader = BaseConfig(
            "jsonFile",
            {"metadata": self._keys_to_json(items), "separateByObject": True},
        )
        config = UpsertConfig(
            loader=loader.to_dict(),
            embedding=self.embedding,
            vectorStore=vector_store,
            recordManager=record_manager,
            loaderName="know-everything-ai",
            docStore={"name": ds.name},
        )
        files = {
            "files": (
                "knowledge.json",
                json.dumps(items, ensure_ascii=False).encode(),
                "application/json",
            )
        }
        return await self.flowise.document_store.upsert_document(
            store_id=ds.id, config=config, files=files
        )

    @staticmethod
    def _keys_to_json(rows: list[dict[str, Any]]) -> str:
        """Map every distinct key except ``content`` to a JSON pointer.

        Flowise needs this to know which object fields are metadata and which
        is the text that gets embedded.
        """
        keys = {
            key
            for row in rows
            if isinstance(row, dict)
            for key in row
            if key != "content"
        }
        return json.dumps({key: f"/{key}" for key in sorted(keys)}, ensure_ascii=False)


def settings_backend(settings: Settings) -> str:
    """Human-readable name of the configured tokenizer backend."""
    if settings.TOKENIZER_BACKEND == "heuristic":
        return "heuristic"
    if settings.TOKENIZER_ENCODING:
        return f"tiktoken:{settings.TOKENIZER_ENCODING}"
    return "tiktoken:auto"
