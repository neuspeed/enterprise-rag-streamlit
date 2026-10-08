import json
from typing import Any

from .embeddings import EmbeddingConfig
from .record_manager import RecordManagerConfig
from .vector_store import VectorStoreConfig


class DocumentStore:
    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        loaders: str | None = None,
        whereUsed: str | None = None,
        status: str | None = None,
        vectorStoreConfig: str | None = None,
        embeddingConfig: str | None = None,
        recordManagerConfig: str | None = None,
        createdDate: str | None = None,
        updatedDate: str | None = None,
        **kwargs
    ):
        self.id = id
        self.name = name
        self.description = description
        self.loaders = loaders
        self.whereUsed = whereUsed
        self.status = status
        self.vectorStoreConfig = vectorStoreConfig
        self.embeddingConfig = embeddingConfig
        self.recordManagerConfig = recordManagerConfig
        self.createdDate = createdDate
        self.updatedDate = updatedDate

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentStore":
        return cls(**data)


class DocumentChunk:
    def __init__(
        self,
        id: str | None = None,
        docId: str | None = None,
        storeId: str | None = None,
        chunkNo: int | None = None,
        pageContent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id
        self.docId = docId
        self.storeId = storeId
        self.chunkNo = chunkNo
        self.pageContent = pageContent
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.pageContent is not None:
            result["pageContent"] = self.pageContent
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class DocumentFile:
    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        mimePrefix: str | None = None,
        size: int | None = None,
        status: str | None = None,
        uploaded: str | None = None,
    ):
        self.id = id
        self.name = name
        self.mimePrefix = mimePrefix
        self.size = size
        self.status = status
        self.uploaded = uploaded


class DocumentLoader:
    def __init__(
        self,
        id: str | None = None,
        loaderId: str | None = None,
        loaderName: str | None = None,
        loaderConfig: dict[str, Any] | None = None,
        splitterId: str | None = None,
        splitterName: str | None = None,
        splitterConfig: dict[str, Any] | None = None,
        totalChunks: int | None = None,
        totalChars: int | None = None,
        status: str | None = None,
        storeId: str | None = None,
        files: list[DocumentFile] | None = None,
        source: str | None = None,
        credential: str | None = None,
        rehydrated: bool | None = None,
        preview: bool | None = None,
        previewChunkCount: int | None = None,
    ):
        self.id = id
        self.loaderId = loaderId
        self.loaderName = loaderName
        self.loaderConfig = loaderConfig or {}
        self.splitterId = splitterId
        self.splitterName = splitterName
        self.splitterConfig = splitterConfig or {}
        self.totalChunks = totalChunks
        self.totalChars = totalChars
        self.status = status
        self.storeId = storeId
        self.files = files or []
        self.source = source
        self.credential = credential
        self.rehydrated = rehydrated
        self.preview = preview
        self.previewChunkCount = previewChunkCount


class ChunkResponse:
    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        count: int | None = None,
        file: DocumentLoader | None = None,
        currentPage: int | None = None,
        storeName: str | None = None,
        description: str | None = None,
    ):
        self.chunks = chunks or []
        self.count = count
        self.file = file
        self.currentPage = currentPage
        self.storeName = storeName
        self.description = description

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkResponse":
        chunks = [DocumentChunk.from_dict(c) for c in data.get("chunks", [])]
        file_data = data.get("file")
        file = DocumentLoader(**file_data) if file_data else None
        return cls(
            chunks=chunks,
            count=data.get("count"),
            file=file,
            currentPage=data.get("currentPage"),
            storeName=data.get("storeName"),
            description=data.get("description"),
        )


class UpsertConfig:
    """Configuration for ``POST /document-store/upsert/:id``.

    Callers pass typed objects. Flowise's endpoint takes this as a multipart
    form whose nested sections are JSON *strings*, so the serialisation happens
    in :meth:`to_dict` — previously the pipeline had to ``json.dumps`` each
    piece by hand, which is how an ``embedding`` object ended up being passed
    where a ``dict`` was declared.
    """

    #: Fields sent to Flowise as embedded JSON rather than as plain form values.
    JSON_FIELDS = (
        "metadata",
        "docStore",
        "loader",
        "splitter",
        "embedding",
        "vectorStore",
        "recordManager",
    )

    def __init__(
        self,
        docId: str | None = None,
        metadata: dict[str, Any] | None = None,
        replaceExisting: bool | None = None,
        createNewDocStore: str | None = None,
        docStore: dict[str, Any] | None = None,
        loader: dict[str, Any] | None = None,
        splitter: dict[str, Any] | None = None,
        embedding: EmbeddingConfig | dict[str, Any] | None = None,
        vectorStore: VectorStoreConfig | dict[str, Any] | None = None,
        recordManager: RecordManagerConfig | dict[str, Any] | None = None,
        loaderName: str | None = None,
    ):
        self.docId = docId
        self.metadata = metadata
        self.replaceExisting = replaceExisting
        self.createNewDocStore = createNewDocStore
        self.docStore = docStore
        self.loader = loader
        self.splitter = splitter
        self.embedding = embedding
        self.vectorStore = vectorStore
        self.recordManager = recordManager
        self.loaderName = loaderName

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            if key in self.JSON_FIELDS:
                value = json.dumps(value, ensure_ascii=False)
            result[key] = value
        return result


class UpsertResult:
    def __init__(
        self,
        numAdded: int | None = None,
        numDeleted: int | None = None,
        numUpdated: int | None = None,
        numSkipped: int | None = None,
        addedDocs: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        self.numAdded = numAdded
        self.numDeleted = numDeleted
        self.numUpdated = numUpdated
        self.numSkipped = numSkipped
        self.addedDocs = addedDocs or []


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UpsertResult":
        return cls(**data)


class RetrievalQuery:
    def __init__(self, storeId: str, query: str):
        self.storeId = storeId
        self.query = query

    def to_dict(self) -> dict[str, Any]:
        return {"storeId": self.storeId, "query": self.query}


class RetrievalResult:
    def __init__(
        self,
        timeTaken: int | None = None,
        docs: list[dict[str, Any]] | None = None,
    ):
        self.timeTaken = timeTaken
        self.docs = docs or []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        return cls(**data)


class RefreshItem:
    def __init__(
        self,
        docId: str | None = None,
        metadata: dict[str, Any] | None = None,
        replaceExisting: bool | None = None,
        createNewDocStore: bool | None = None,
        docStore: dict[str, Any] | None = None,
        loader: dict[str, Any] | None = None,
        splitter: dict[str, Any] | None = None,
        embedding: EmbeddingConfig | dict[str, Any] | None = None,
        vectorStore: VectorStoreConfig | dict[str, Any] | None = None,
        recordManager: RecordManagerConfig | dict[str, Any] | None = None,
    ):
        self.docId = docId
        self.metadata = metadata
        self.replaceExisting = replaceExisting
        self.createNewDocStore = createNewDocStore
        self.docStore = docStore
        self.loader = loader
        self.splitter = splitter
        self.embedding = embedding
        self.vectorStore = vectorStore
        self.recordManager = recordManager

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if hasattr(value, "to_dict"):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result
