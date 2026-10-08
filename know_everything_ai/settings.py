"""Runtime configuration for know-everything-ai.

Every value is overridable from the environment. No secret, host or credential
is baked in: an unset secret stays empty and is reported by
:meth:`Settings.missing_required`, which the CLI turns into a startup error.
That keeps the package importable in tests and tooling while still failing fast
for a real deployment.

Attribute access is intentionally flat. Nested settings groups would touch every
call site for no functional gain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from know_everything_ai.prompts import (
    ANSWER_PROMPT,
    CHUNK_CLASSIFIER_PROMPT,
    KNOWLEDGE_CANVAS_PROMPT,
    PDF_EXTRACTION_PROMPT,
)


def _discover_env_file() -> str | None:
    """Locate the ``.env`` file for both a source checkout and a container.

    In a container the package sits at ``/app/know_everything_ai`` and the env
    file at ``/app/.env``. In a checkout it sits at ``<repo>/.env``. Both are the
    parent of the package directory, so one candidate covers each; the CWD
    fallback covers unusual layouts.
    """
    package_dir = Path(__file__).resolve().parent
    for candidate in (
        package_dir.parent / ".env",
        package_dir.parent.parent / ".env",
        Path.cwd() / ".env",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_discover_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        # Compose writes `FOO: ${FOO:-}` for every setting so an operator can
        # override one. Those lines export an empty string when the variable is
        # unset, and pydantic-settings treats an empty variable as a real value
        # that overrides the field default — so a stock deploy would start with
        # LOCAL_EMBEDDING_MODEL="" and fail inside fastembed instead of using
        # the default written next to it. Empty means "not set" throughout.
        env_ignore_empty=True,
    )

    # ------------------------------------------------------------------ runtime
    ENV: Literal["dev", "stage", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    APP_NAME: str = "know-everything-ai"

    # -------------------------------------------------------------------- paths
    DATA_DIR: Path = Field(default=Path("./data"))

    # ------------------------------------------------------- limits and quotas
    # A self-hosted buyer typically runs this on a small VPS. These caps exist so
    # a single oversized spreadsheet cannot take the box down: the table parser
    # loads an entire .xlsx into pandas.
    MAX_UPLOAD_MB: int = 100
    MAX_FILES_PER_KB: int = 50
    MAX_DOCUMENTS_PER_KB: int = 500
    MAX_CONCURRENT_JOBS: int = 2
    # Fetching arbitrary URLs is an SSRF vector. Keep this off unless every
    # source host is trusted.
    ALLOW_PRIVATE_URLS: bool = False
    ALLOW_LOCAL_SOURCES: bool = False

    # ------------------------------------------------- generation: knowledge canvas
    ENRICHER_API_URL: str = ""
    ENRICHER_API_KEY: str = ""
    ENRICHER_MODEL_NAME: str = ""
    ENRICHER_PROMPT: str = KNOWLEDGE_CANVAS_PROMPT
    ENRICHER_TIMEOUT: int = 1800
    ENRICHER_MAX_TOKENS: int | None = None
    # Second provider for the same call, tried when the first rate-limits,
    # times out or returns a 5xx. Usually a local Ollama: same job, no per-token
    # cost, no provider quota. A model name is required with a URL because the
    # fallback rarely runs the same weights as the primary.
    ENRICHER_FALLBACK_API_URL: str = ""
    ENRICHER_FALLBACK_API_KEY: str = ""
    ENRICHER_FALLBACK_MODEL_NAME: str = ""

    # ------------------------------------------------------ generation: classifier
    CLASSIFIER_API_URL: str = ""
    CLASSIFIER_API_KEY: str = ""
    CLASSIFIER_MODEL_NAME: str = ""
    CLASSIFIER_PROMPT_TEMPLATE: str = CHUNK_CLASSIFIER_PROMPT
    # The classifier is the cheapest call in the pipeline and the one most
    # likely to be handed to a small local model, so it is the usual first
    # fallback target.
    CLASSIFIER_FALLBACK_API_URL: str = ""
    CLASSIFIER_FALLBACK_API_KEY: str = ""
    CLASSIFIER_FALLBACK_MODEL_NAME: str = ""
    CLASSIFIER_CATEGORY_DESCRIPTIONS: dict[str, str] = {
        "faq": "explicit question-answer pair (e.g. 'Вопрос: ... Ответ: ...')",
        "glossary": "term and its definition",
        "document_chunk": "regular text (instruction, description, paragraph)",
        "table": "table (rows and columns)",
        "list": "bulleted or numbered list",
        "code": "code block or commands",
    }
    # Fragments shorter than this are dropped without calling the model.
    CLASSIFIER_TRASH_SIZE: int = 200

    # ------------------------------------------------------------- shared LLM knobs
    LLM_TIMEOUT: int = 600
    LLM_MAX_RETRIES: int = 3

    # ------------------------------------------------------------ VLM: PDF parsing
    VLM_API_URL: str = ""
    VLM_API_KEY: str = "not-needed"
    VLM_MODEL_NAME: str = ""
    VLM_PROMPT_TEMPLATE: str = PDF_EXTRACTION_PROMPT
    VLM_TIMEOUT: int = 1800
    VLM_FALLBACK_API_URL: str = ""
    VLM_FALLBACK_API_KEY: str = ""
    VLM_FALLBACK_MODEL_NAME: str = ""
    MAX_CONCURRENT_PDFS: int = 2
    # All pages of a PDF are sent to the model as images in one request. Beyond
    # this they are split into batches so a 200-page scan cannot exceed the
    # request limit.
    PDF_MAX_PAGES_PER_REQUEST: int = 20
    PDF_RENDER_DPI: int = 200
    POPPLER_PATH: str = "/usr/bin"

    # ---------------------------------------------------- agent answer role (read side)
    # Answering is deliberately a separate role from the extraction prompts: a
    # plain retrieval call must never spend model tokens, and the assistant the
    # buyer deploys deserves its own model and prompt rather than inheriting the
    # enricher's. All three are optional at startup — a retrieval-only deployment
    # asks why it earned a 503 only when it actually sends a question, not when
    # it starts.
    ANSWER_API_URL: str = ""
    ANSWER_API_KEY: str = ""
    ANSWER_MODEL_NAME: str = "glm-5.3-flash"
    ANSWER_SYSTEM_PROMPT: str = ANSWER_PROMPT
    ANSWER_TIMEOUT: int = 120
    ANSWER_MAX_TOKENS: int | None = 1200
    ANSWER_TEMPERATURE: float = 0.2
    # How many fragments a single answer may cite.
    ANSWER_MAX_FRAGMENTS: int = 6
    ANSWER_FALLBACK_API_URL: str = ""
    ANSWER_FALLBACK_API_KEY: str = ""
    ANSWER_FALLBACK_MODEL_NAME: str = ""

    # ------------------------------------------------------------------ structurizer
    STRUCTURIZER_USE_LLM: bool = False
    STRUCTURIZER_PROMPT: str = ""
    STRUCTURIZER_TIMEOUT: int = 300

    # -------------------------------------------------- branching and tokenization
    # Documents at or below this size take the context-knowledge-base branch;
    # larger ones are chunked, classified and structured into a vector store.
    MAX_CKB_TOKENS: int = 220_000
    TOKENIZER_BACKEND: Literal["auto", "tiktoken", "heuristic"] = "auto"
    TOKENIZER_ENCODING: str = ""
    # Calibrated against cl100k_base by measuring character counts per token on
    # Russian, English and mixed prose; the two values are blended by the share
    # of Cyrillic letters rather than switched by a boolean. On a 800k-character
    # Russian corpus this lands within 0.7% of the true count. The inherited
    # heuristic used 3.2 for Cyrillic and was 38% low, which is the difference
    # between picking the context branch and overflowing the enricher's context
    # window. Changing these numbers invalidates the golden tokenizer test.
    TOKENIZER_RU_CHARS_PER_TOKEN: float = 2.35
    TOKENIZER_EN_CHARS_PER_TOKEN: float = 4.6

    # ------------------------------------------------------------------- chunking
    CHUNK_SIZE_TOKENS: int = 800
    CHUNK_OVERLAP_TOKENS: int = 100

    # --------------------------------------------------------- source fetching
    SOURCE_HTTP_TIMEOUT: int = 60
    SOURCE_MAX_REDIRECTS: int = 5
    # Optional WebDAV mount (Nextcloud, ownCloud, ...) for sources that need auth.
    WEBDAV_BASE_URL: str = ""
    WEBDAV_USER: str = ""
    WEBDAV_PASSWORD: str = ""

    # ----------------------------------------------- Flowise (optional adapter)
    FLOWISE_ENABLED: bool = False
    FLOWISE_API_URL: str = ""
    FLOWISE_API_KEY: str = ""
    FLOWISE_TIMEOUT: float = 60.0
    # Ingestion runs inside Flowise, so an upsert legitimately outlives the
    # ordinary request timeout on a store with many chunks.
    FLOWISE_UPSERT_TIMEOUT: float = 600.0

    EMBEDDING_API_URL: str = ""
    EMBEDDING_MODEL: str = ""
    EMBEDDING_CREDENTIAL: str = ""

    # Local embeddings for the built-in pgvector store (ONNX, no GPU, no API).
    #
    # Deliberately separate from EMBEDDING_* above: those configure the provider
    # Flowise calls, and folding the two together would silently repoint the
    # Flowise upsert at a model name Flowise cannot resolve.
    #
    # paraphrase-multilingual-MiniLM-L12-v2 rather than PLAN's
    # intfloat/multilingual-e5-small: the latter is not in fastembed's supported
    # list and needs manual registration plus its query:/passage: prefix
    # convention to behave. This one ships in the list, covers Russian, and takes
    # 0.22 GB instead of 2.24 GB for the large variant.
    LOCAL_EMBEDDING_MODEL: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # Baked into the vector column type at migration time. Changing it means
    # re-embedding everything, so a mismatch with what is in the database is an
    # error rather than a silent quality drop.
    LOCAL_EMBEDDING_DIM: int = 384
    LOCAL_EMBEDDING_BATCH_SIZE: int = 32
    # e5-family models rank by different conventions for a search string and a
    # document, e.g. "query: " / "passage: ". The default MiniLM does not, so
    # these start empty. Changing LOCAL_EMBEDDING_MODEL to an e5 variant without
    # setting them degrades ranking with no error anywhere on that path.
    LOCAL_EMBEDDING_QUERY_PREFIX: str = ""
    LOCAL_EMBEDDING_DOC_PREFIX: str = ""
    # Empty means fastembed's own default cache. Set it to a mounted volume in
    # containers or the first query blocks on a download that `down -v` then
    # repeats.
    FASTEMBED_CACHE_DIR: str = ""

    VECTOR_STORE_HOST: str = "localhost"
    VECTOR_STORE_PORT: int = 5432
    VECTOR_STORE_USER: str = "postgres"
    VECTOR_STORE_DATABASE: str = "vector_db"
    VECTOR_STORE_CREDENTIAL: str = ""
    VECTOR_STORE_TOPK: int = 10

    # ------------------------------------------------ query API (read side only)
    # Retrieval ranks and cites fragments; it never spends a model token.
    # Answering is the agent role, configured separately, so a plain query
    # cannot be made expensive by a setting meant for someone else's prompts.
    QUERY_API_HOST: str = "0.0.0.0"
    QUERY_API_PORT: int = 8000
    # A caller asking for more than this is clamped rather than rejected: a
    # client that requests 1000 hits should get the best 50, not a 422.
    QUERY_MAX_LIMIT: int = 50
    # Cosine score floor. -1.0 disables filtering: the score is 1 minus the
    # cosine distance, so it can legitimately be negative for an unrelated
    # fragment and a default of 0.0 would silently drop results.
    QUERY_MIN_SCORE: float = -1.0
    # How long a verified key stays in memory. Revocation therefore takes
    # effect within this window, which is the trade against hitting the
    # database on every request.
    QUERY_API_KEY_CACHE_SECONDS: int = 30

    RECORD_MANAGER_HOST: str = "localhost"
    RECORD_MANAGER_PORT: int = 5432
    RECORD_MANAGER_DATABASE: str = "records_db"
    RECORD_MANAGER_CREDENTIAL: str = ""

    # --------------------------------------------------- callback webhook (optional)
    WEBHOOK_TIMEOUT: int = 30
    WEBHOOK_MAX_RETRIES: int = 3

    # ------------------------------------------------------------------ transport
    INGEST_TRANSPORT: Literal["postgres", "rabbitmq"] = "postgres"
    MQ_HOST: str = ""
    MQ_PORT: int = 5672
    MQ_USER: str = ""
    MQ_PASS: str = ""
    MQ_VIRTUAL_HOST: str = "/"
    MQ_EXCHANGE: str = "{ENV}_knowledge_base_tasks"
    MQ_DLX_EXCHANGE: str = "dlx.knowledge.{ENV}"
    MQ_DLQ_QUEUE: str = "{ENV}_knowledge_base_tasks.dlq"
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 60
    GLOBAL_CONCURRENCY: int = 4
    CLIENT_WEIGHTS: dict[str, int] = Field(default_factory=dict)

    # ----------------------------------------------------------------- job runner
    MESSAGE_PROCESS_TIMEOUT_SECONDS: int = 1800
    SHUTDOWN_DRAIN_TIMEOUT_SECONDS: int = 25

    # ----------------------------------------------------------------- validators
    @field_validator(
        "ENRICHER_API_URL",
        "CLASSIFIER_API_URL",
        "VLM_API_URL",
        "ANSWER_API_URL",
        "ENRICHER_FALLBACK_API_URL",
        "CLASSIFIER_FALLBACK_API_URL",
        "VLM_FALLBACK_API_URL",
        "ANSWER_FALLBACK_API_URL",
        "FLOWISE_API_URL",
        "EMBEDDING_API_URL",
        "WEBDAV_BASE_URL",
        mode="after",
    )
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/") if value else value

    @field_validator(
        "ENRICHER_FALLBACK_MODEL_NAME",
        "CLASSIFIER_FALLBACK_MODEL_NAME",
        "VLM_FALLBACK_MODEL_NAME",
        "ANSWER_FALLBACK_MODEL_NAME",
        mode="after",
    )
    @classmethod
    def _fallback_model_needs_url(cls, value: str, info) -> str:
        """A model name without a URL would configure failover that never runs.

        Rejecting it at startup beats discovering during an incident that the
        fallback was never wired up: the job looks configured to fail over and
        does not.
        """
        if value and not info.data.get(f"{info.field_name.removesuffix('_MODEL_NAME')}_API_URL"):
            raise ValueError(
                f"{info.field_name} is set but the matching "
                f"{info.field_name.removesuffix('_MODEL_NAME')}_API_URL is empty; "
                "the fallback would never be used"
            )
        return value

    def database_dsn(self) -> str:
        """Postgres connection string built from the VECTOR_STORE_* settings.

        Built rather than added as its own variable so a deployment only has one
        place to change a host and the credentials stay out of the composed
        string's history.
        """
        # Quoted: a generated password carrying '@' or ':' would otherwise
        # shift the DSN's host out of the authority section and connect nowhere.
        user = quote(self.VECTOR_STORE_USER or "postgres", safe="")
        password = quote(self.VECTOR_STORE_CREDENTIAL, safe="")
        auth = f"{user}:{password}@" if password else f"{user}@"
        host = self.VECTOR_STORE_HOST
        port = self.VECTOR_STORE_PORT
        return f"postgresql://{auth}{host}:{port}/{self.VECTOR_STORE_DATABASE}"

    @field_validator("LOG_LEVEL")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unsupported LOG_LEVEL: {value!r}")
        return level

    # ------------------------------------------------------------------- required
    def missing_required(
        self, role: Literal["worker", "api"] = "worker"
    ) -> list[str]:
        """Return the settings a real deployment needs but has not been given.

        Split by role because the two processes really do differ: the query API
        only reads vectors, and requiring a vision-model URL from it would hold
        a read-only service down over a setting that belongs to ingestion.
        """
        missing: list[str] = []

        def require(name: str, value: str) -> None:
            if not value.strip():
                missing.append(name)

        if role == "api":
            require("VECTOR_STORE_HOST", self.VECTOR_STORE_HOST)
            require("VECTOR_STORE_DATABASE", self.VECTOR_STORE_DATABASE)
            return missing

        require("ENRICHER_API_URL", self.ENRICHER_API_URL)
        require("ENRICHER_MODEL_NAME", self.ENRICHER_MODEL_NAME)
        require("CLASSIFIER_API_URL", self.CLASSIFIER_API_URL)
        require("CLASSIFIER_MODEL_NAME", self.CLASSIFIER_MODEL_NAME)
        require("VLM_API_URL", self.VLM_API_URL)
        require("VLM_MODEL_NAME", self.VLM_MODEL_NAME)

        if self.INGEST_TRANSPORT == "rabbitmq":
            require("MQ_HOST", self.MQ_HOST)
            require("MQ_USER", self.MQ_USER)
            require("MQ_PASS", self.MQ_PASS)
            # Without a client the dispatcher declares no queues, binds nothing
            # and starts cleanly, so the worker sits idle and healthy while
            # consuming nothing. Refusing to start is the only visible symptom.
            if not self.CLIENT_WEIGHTS:
                missing.append("CLIENT_WEIGHTS")

        if self.FLOWISE_ENABLED:
            require("FLOWISE_API_URL", self.FLOWISE_API_URL)
            require("FLOWISE_API_KEY", self.FLOWISE_API_KEY)

        return missing

    def ensure_valid(self, role: Literal["worker", "api"] = "worker") -> None:
        """Raise if a required setting is absent.

        Called by the entrypoint before any network connection is opened, so a
        misconfigured deployment fails at startup with a list of what to set
        rather than coming up and quietly doing nothing.
        """
        missing = self.missing_required(role)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing required configuration: {joined}. "
                f"Set them in the environment or in .env "
                f"(see .env.example), then restart."
            )
