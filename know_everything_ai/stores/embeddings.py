"""Local embeddings via fastembed (ONNX, CPU, no API key).

Two properties drive the shape of this module.

The library is synchronous — ``TextEmbedding.embed`` returns a generator of
numpy arrays with no async variant — so every call is pushed onto a worker
thread. Embedding a batch inline would block the event loop that the pipeline
shares with concurrent jobs, which is the same class of stall the parsers were
moved off for.

The weights are downloaded on first use. ``cache_dir`` points at a mounted
volume in a container; without it a rebuild re-downloads the model and the first
request of the day pays for it. :meth:`warm` exists so that cost lands at
startup instead of in a buyer's first query.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import structlog

from know_everything_ai.settings import Settings

log = structlog.get_logger("embeddings")


class FastembedEmbedder:
    """Text embeddings computed locally.

    ``model`` and ``dim`` are read from settings rather than probed from the
    loaded model so that a mismatch with the database column is detectable
    before anything is written.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.LOCAL_EMBEDDING_MODEL
        self._dim = settings.LOCAL_EMBEDDING_DIM
        self._batch_size = settings.LOCAL_EMBEDDING_BATCH_SIZE
        self._cache_dir = settings.FASTEMBED_CACHE_DIR or None
        # The e5 family expects these; the default MiniLM does not, so the
        # defaults are empty. Swapping the model without swapping the prefixes
        # would quietly rank by the wrong convention, and no exception is raised
        # anywhere along that path.
        self._query_prefix = settings.LOCAL_EMBEDDING_QUERY_PREFIX
        self._doc_prefix = settings.LOCAL_EMBEDDING_DOC_PREFIX
        self._model = None
        self._load_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    async def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                log.info(
                    "embedding_model_loading",
                    model=self._model_name,
                    cache_dir=self._cache_dir,
                )
                self._model = await asyncio.to_thread(self._load)
        return self._model

    def _load(self):
        from fastembed import TextEmbedding

        return TextEmbedding(
            model_name=self._model_name,
            cache_dir=self._cache_dir,
        )

    async def warm(self) -> None:
        """Download and initialise now so the first request is not the one that
        pays for it."""
        await self._ensure_loaded()
        log.info("embedding_model_ready", model=self._model_name, dim=self._dim)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_loaded()
        prefixed = [f"{self._doc_prefix}{t}" for t in texts]

        def run() -> list[list[float]]:
            vectors = model.embed(
                prefixed, batch_size=max(1, self._batch_size)
            )
            return [vector.tolist() for vector in vectors]

        result = await asyncio.to_thread(run)
        self._verify_dim(result)
        return result

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed_documents([f"{self._query_prefix}{text}"])
        if not result:
            raise RuntimeError("embedding model returned no vector for the query")
        return result[0]

    def _verify_dim(self, vectors: list[list[float]]) -> None:
        if vectors and len(vectors[0]) != self._dim:
            raise ValueError(
                f"Embedding model {self._model_name!r} returned "
                f"{len(vectors[0])}-dimensional vectors but "
                f"LOCAL_EMBEDDING_DIM is {self._dim}. The column type was "
                "created for the configured dimension, so storing these would "
                "either fail or silently truncate."
            )

    async def aclose(self) -> None:
        # onnxruntime releases its session on garbage collection; there is no
        # close hook to call, and dropping the reference is what frees it.
        self._model = None


class FakeEmbedder:
    """Deterministic stand-in for tests.

    A hashed vector keeps two different texts at a non-zero distance, so ranking
    assertions stay meaningful without a 220 MB model in the test run.
    """

    def __init__(self, dim: int = 384, model: str = "fake/embedder") -> None:
        self._dim = dim
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        seed = abs(hash(text)) or 1
        return [
            (((seed >> (index % 24)) ^ index * 2654435761) % 1000) / 1000.0
            for index in range(self._dim)
        ]

    async def aclose(self) -> None:
        return None
