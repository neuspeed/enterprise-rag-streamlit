from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from .embeddings import EmbeddingConfig
from .vector_store import VectorStoreConfig
from .record_manager import RecordManagerConfig


class DocumentStore:
    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        loaders: Optional[str] = None,
        whereUsed: Optional[str] = None,
        status: Optional[str] = None,
        vectorStoreConfig: Optional[str] = None,
        embeddingConfig: Optional[str] = None,
        recordManagerConfig: Optional[str] = None,
        createdDate: Optional[str] = None,
        updatedDate: Optional[str] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStore":
        return cls(**data)


class DocumentChunk:
    def __init__(
        self,
        id: Optional[str] = None,
        docId: Optional[str] = None,
        storeId: Optional[str] = None,
        chunkNo: Optional[int] = None,
        pageContent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.docId = docId
        self.storeId = storeId
        self.chunkNo = chunkNo
        self.pageContent = pageContent
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.pageContent is not None:
            result["pageContent"] = self.pageContent
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class DocumentFile:
    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        mimePrefix: Optional[str] = None,
        size: Optional[int] = None,
        status: Optional[str] = None,
        uploaded: Optional[str] = None,
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
        id: Optional[str] = None,
        loaderId: Optional[str] = None,
        loaderName: Optional[str] = None,
        loaderConfig: Optional[Dict[str, Any]] = None,
        splitterId: Optional[str] = None,
        splitterName: Optional[str] = None,
        splitterConfig: Optional[Dict[str, Any]] = None,
        totalChunks: Optional[int] = None,
        totalChars: Optional[int] = None,
        status: Optional[str] = None,
        storeId: Optional[str] = None,
        files: Optional[List[DocumentFile]] = None,
        source: Optional[str] = None,
        credential: Optional[str] = None,
        rehydrated: Optional[bool] = None,
        preview: Optional[bool] = None,
        previewChunkCount: Optional[int] = None,
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
        chunks: Optional[List[DocumentChunk]] = None,
        count: Optional[int] = None,
        file: Optional[DocumentLoader] = None,
        currentPage: Optional[int] = None,
        storeName: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.chunks = chunks or []
        self.count = count
        self.file = file
        self.currentPage = currentPage
        self.storeName = storeName
        self.description = description

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkResponse":
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
    def __init__(
        self,
        docId: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        replaceExisting: Optional[bool] = None,
        createNewDocStore: Optional[str] = None,
        docStore: Optional[Dict[str, Any]] = None,
        loader: Optional[Dict[str, Any]] = None,
        splitter: Optional[Dict[str, Any]] = None,
        embedding: Optional[Union[EmbeddingConfig, Dict[str, Any]]] = None,
        vectorStore: Optional[Union[VectorStoreConfig, Dict[str, Any]]] = None,
        recordManager: Optional[Union[RecordManagerConfig, Dict[str, Any]]] = None,
        loaderName: Optional[str] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if hasattr(value, "to_dict"):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result


class UpsertResult:
    def __init__(
        self,
        numAdded: Optional[int] = None,
        numDeleted: Optional[int] = None,
        numUpdated: Optional[int] = None,
        numSkipped: Optional[int] = None,
        addedDocs: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        self.numAdded = numAdded
        self.numDeleted = numDeleted
        self.numUpdated = numUpdated
        self.numSkipped = numSkipped
        self.addedDocs = addedDocs or []
        

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpsertResult":
        return cls(**data)


class RetrievalQuery:
    def __init__(self, storeId: str, query: str):
        self.storeId = storeId
        self.query = query

    def to_dict(self) -> Dict[str, Any]:
        return {"storeId": self.storeId, "query": self.query}


class RetrievalResult:
    def __init__(
        self,
        timeTaken: Optional[int] = None,
        docs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.timeTaken = timeTaken
        self.docs = docs or []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        return cls(**data)


class RefreshItem:
    def __init__(
        self,
        docId: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        replaceExisting: Optional[bool] = None,
        createNewDocStore: Optional[bool] = None,
        docStore: Optional[Dict[str, Any]] = None,
        loader: Optional[Dict[str, Any]] = None,
        splitter: Optional[Dict[str, Any]] = None,
        embedding: Optional[Union[EmbeddingConfig, Dict[str, Any]]] = None,
        vectorStore: Optional[Union[VectorStoreConfig, Dict[str, Any]]] = None,
        recordManager: Optional[Union[RecordManagerConfig, Dict[str, Any]]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if hasattr(value, "to_dict"):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result