# MVP Plan — Knowledge Base Builder

Sellable, self-hosted service. A buyer runs `docker compose up`, uploads documents (or URLs /
pasted text), and the service automatically produces either:

- a **context KB** — one self-contained JSON knowledge canvas, exportable, ready to feed a long-context model; or
- a **vector KB** — chunk → classify → structure → pgvector, searchable with citations.

The buyer never chooses a chunking strategy. The service branches on measured size.

---

## 1. Locked decisions

### Naming

| Thing | Value |
|---|---|
| Product | **know-everything-ai** |
| Distribution (PyPI) | `know-everything-ai` |
| Python package | `know_everything_ai` |
| CLI | `know-everything-ai serve` / `worker` / `migrate` |
| Docker image / compose service | `know-everything-ai` |
| Flowise document store | `kb_<kb_external_id>` (unchanged contract) |
| API keys | `Authorization: Bearer <key>` |

### Product decisions

| Area | Decision |
|---|---|
| Product form | Self-hosted `docker compose up`. Small B2B buyer, single-tenant-friendly. |
| Vector store | pgvector built-in default behind a `VectorStore` interface. Flowise = optional adapter. |
| Generation models | Bring-your-own OpenAI-compatible endpoint (vLLM / Ollama / LM Studio / Together / OpenAI). |
| Embeddings | Local `fastembed` `intfloat/multilingual-e5-small` default (Russian-dominant corpus), OpenAI-compatible override. |
| Hero feature | Both branches; vector/RAG path is the polished demo and has the query endpoint. |
| License | Proprietary EULA, per-legal-entity, per-deployment `LICENSE_KEY`, priced per deployment + support tier. |
| Auth | Email + password session cookies for the UI; API keys for machines. Registration off by default. |
| Tenancy | Shared tables + `kb_id` filtering + Postgres Row-Level Security. `kb_id` namespace on both stores. |
| Job queue | Postgres `FOR UPDATE SKIP LOCKED`. RabbitMQ retained behind `INGEST_TRANSPORT=rabbitmq` as a legacy adapter. |
| TLS / packaging | None. API binds `:8000`; buyer's own reverse proxy in front. No air-gapped installer. |

### Deliberately not built

Chat widget · Stripe / per-token billing · web crawling · ACL & roles · air-gapped install ·
horizontal scale · RabbitMQ on the default path · `schema`-per-tenant.

---

## 2. Known defects being fixed in P0

These are verified on the critical path — each crashes or stalls a real job.

| # | Defect | Location |
|---|---|---|
| B1 | `DocCategory.DOCUMENT_CHUNK` does not exist (member is `DOCUMENT`) → `AttributeError` on any markdown code block | `know_everything_ai/utils/markdown_to_raw_elements.py` |
| B2 | `backoff` with no `max_tries` + sync `requests` inside the event loop → infinite retry stalls every in-flight message | `know_everything_ai/utils/webhook_utils.py` |
| B3 | `headers` unbound when `api_key` is empty → `UnboundLocalError` | `know_everything_ai/flowise/client.py`, `know_everything_ai/flowise/resources/document_store.py` |
| B4 | `element` used before assignment → duplicate `RawElement`s for every paragraph nested in a docx table | `know_everything_ai/parsers/word_parser.py` |
| B5 | `print(e); return None` swallows all HTML load failures; `__init__` never calls `super()`, so `self.owncloud_client` never exists | `know_everything_ai/loaders/html_loader.py` |

Also addressed in P0:

- All Flowise calls are sync `requests` invoked from async code.
- A new `KnowledgePipeline` (and a new `AsyncOpenAI` httpx pool) is constructed **per message** — fd leak.
- Classifier's catch-all fallback marks unknown chunks `is_useful=True`, so garbage reaches the upsert.
- `cost={}` is never populated.
- Production credentials and internal hosts hardcoded in `settings.py`.
- Tests: 48 passed / 17 failed / 1 collection error, no pytest config, no `pytest-asyncio`.

### Russian-corpus tokenizer bug

`requirements.txt` has no `tiktoken`, so `estimate_tokens` always falls back to its heuristic:
3.2 chars/token for Cyrillic. Real Russian in `cl100k_base` is ≈2 chars/token, so the heuristic
**underestimates by ~35–40%**. Since `MAX_CKB_TOKENS` gates the Small/Large decision, a document
estimated at 200k can actually be 280k real tokens and overflow the enricher context window.

Fix: add `tiktoken`; `TOKENIZER_BACKEND = auto|tiktoken|heuristic`; `auto` uses
`encoding_for_model()` when the configured model is a known OpenAI model; recalibrate the Russian
coefficient; log which estimator produced the number; pin a fixed Russian sample in a golden test
so the coefficient cannot silently drift.

### Internal-bindings removal

No `infercom` strings, hosts, credentials or CI anywhere. Strip from `settings.py`
(`STAGE_BASIC_AUTH_*`, `PROXY`, hardcoded hosts, all secrets — required env, no defaults),
webhook default URL, `Dockerfile` → `python:3.11-slim` + poppler, `docker-compose.yml` → local
build, `.gitlab-ci.yml` → self-contained, `owncloud_client` → generic `WEBDAV_*` + plain HTTP
download, README rewrite.

---

## 3. Target layout

Flat imports (`from settings import Settings`) are why 4 of 5 test files cannot even be
collected. The restructure is a product prerequisite, not cosmetics.

```
know_everything_ai/                 # renamed from app/ — installable, absolute imports
├── settings.py              # pydantic-settings, nested groups, no secret defaults
├── schemas.py               # RawElement, DocCategory, Payload, JobResult
├── loaders/ parsers/ cleaners/ splitters/ classifier/ structurizers/ enricher/
├── stores/
│   ├── base.py              # VectorStore Protocol: upsert / delete / search / reset
│   ├── pgvector.py          # built-in default
│   └── flowise.py           # optional adapter
├── transports/
│   └── rabbitmq.py          # legacy adapter, opt-in
├── pipeline.py              # branching coordinator, framework-free
├── api/                     # FastAPI (P1)
│   ├── main.py deps.py
│   ├── routers/             # auth, kb, ingest, jobs, query, export, admin, health
│   ├── models/              # public request/response contract
│   └── services/            # ingestion, export, usage
├── worker/                  # job runner: Postgres FOR UPDATE SKIP LOCKED
└── cli.py                   # kb-pipeline serve / worker / migrate
ui/                          # no-build single page, served by the API
migrations/  tests/  docs/
```

Deleted: `server.js`, `index.html`, `package.json` (dev webhook harness), `app/main.py`
(scratch script with an import-time network call), `app/use.py` (executes a live Flowise upsert at
import), `splitters/ml_chunking.py` (dead: zero importers, incompatible schema, asserts enum values
that do not exist), `tests/test_chunk_classifier.py` and `tests/test_knowledge_canvas.py`
(stale: assert removed enum members and a `src` package that never existed).

---

## 4. Phases

### P0 — Preconditions, unblock, de-brand (~4 days)

**Gate: `docker compose up` processes a Russian PDF end-to-end locally, `grep -ri infercom` returns
nothing, full test suite green.**

- `git init` + baseline commit **before** restructuring — a botched refactor must be revertable.
- `app/` → `know_everything_ai/` as an installable package; flat → absolute imports; add missing `__init__.py`.
- pytest config + `pytest-asyncio` + real `conftest.py`; repair the two green suites.
- Fix B1–B5; move blocking parsers to `asyncio.to_thread`; one shared `httpx.AsyncClient` and
  pipeline per worker instead of per message.
- Tokenizer fix + golden test. Classifier fallback → `is_useful=False` with a reason.
- De-brand (see §2). Quotas defined so a small buyer cannot OOM their own box — `table_parser`
  loads a whole `.xlsx` into pandas.

### P1 — API, auth, jobs (~2 weeks)

Tables: `tenants`, `users`, `sessions`, `api_keys`, `knowledge_bases`, `documents`, `jobs`,
`job_events`, `usage_ledger`.

- Email + password, argon2id, **opaque revocable session cookies** (not JWT: revocable in one
  `DELETE`, no secret-rotation story for a self-hosted deploy). Bootstrap admin from env,
  registration disabled by default.
- `POST /v1/knowledge-bases`, `POST .../documents` (upload | URL | raw text), `GET /v1/jobs/{id}`,
  `DELETE .../kb/{id}`, `/healthz`, `/readyz`.
- Postgres `SKIP LOCKED` worker with real failure reasons. Quotas enforced at the API edge.
- CSRF: session cookie + POST → `SameSite=Lax` plus a custom-header check.
- `usage_ledger` is the metering foundation and the future overage lever. Nothing is priced off it.

### P2 — Stores + query (~1 week)

`VectorStore` Protocol, pgvector impl (`kb_id` filtering, composite index
`(kb_id, embedding vector_cosine_ops)`), Flowise adapter. `POST /v1/knowledge-bases/{id}/query`
with citations; `GET .../export` for the context-branch JSON.

Documented scale limit: pgvector HNSW is global, so filtering reduces recall as the table grows.
Ceiling ≈100k chunks per KB with a composite index; beyond that, partition `documents` by `kb_id`.

### P3 — UI + sell (~1 week)

No-build single page: upload/paste, job progress, **branch badge showing token count + which
estimator produced it + the threshold**, payload preview before ingest, search with citations,
export. Then `LICENSE`/EULA, `.env.example`, quickstart README, `LICENSE_KEY` check, sample
Russian corpus, API docs, 5-minute demo script.

**≈7 weeks** for one developer.

---

## 5. Risks

- **PDF→VLM is the slowest, least reliable step** and the first thing a buyer notices. B4/B5 and
  Russian-corpus PDF quality must be validated before promising anything.
- **Multilingual embeddings are load-bearing.** An English-only default silently degrades every
  Russian answer. Pinned in P0, not revisited in P2.
- **Refactor-before-sell carries regression risk.** Mitigation: the two green suites plus a
  golden-file corpus test land in P0 before anything else moves.
- **RabbitMQ adapter keeps a known fd leak** unless ported to the shared-client lifecycle in P0.
