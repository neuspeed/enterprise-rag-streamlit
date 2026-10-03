import requests
from typing import Optional, List, Dict, Any
from ..models import (
    DocumentStore,
    ChunkResponse,
    UpsertConfig,
    UpsertResult,
    RetrievalQuery,
    RetrievalResult,
    RefreshItem,
    DocumentChunk
)


class DocumentStoreResource:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        return headers

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
            if response.status_code == 204:
                return {}
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"API error: {e.response.status_code} - {e.response.text}") from e
    
    def list_document_stores(self) -> List[DocumentStore]:
        url = f"{self.base_url}/document-store/store"
        response = requests.get(url, headers=self._get_headers())
        data = self._handle_response(response)
        return [DocumentStore.from_dict(store) for store in data]
    
    def get_document_store(self, store_id: str) -> DocumentStore:
        url = f"{self.base_url}/document-store/store/{store_id}"
        response = requests.get(url, headers=self._get_headers())
        data = self._handle_response(response)
        return DocumentStore.from_dict(data)
    
    def find_document_store_by_name(self, name: str) -> Optional[DocumentStore]:
        """Ищет Document Store по имени. Возвращает первый совпавший или None."""
        stores: List[DocumentStore] = self.list_document_stores()
        for store in stores:
            if store.name == name:
                return store
        return None

    def get_chunks(self, store_id: str, loader_id: str, page_no: int) -> ChunkResponse:
        url = f"{self.base_url}/document-store/chunks/{store_id}/{loader_id}/{page_no}"
        response = requests.get(url, headers=self._get_headers())
        data = self._handle_response(response)
        return ChunkResponse.from_dict(data)

    def upsert_document(self, store_id: str, config: UpsertConfig, files={}) -> UpsertResult:
        url = f"{self.base_url}/document-store/upsert/{store_id}"
        response = requests.post(url, data=config.to_dict(), headers=self._get_headers(), files=files)
        data = self._handle_response(response)
        return UpsertResult.from_dict(data)

    def refresh_document_store(self, store_id: str, items: List[RefreshItem]) -> List[UpsertResult]:
        url = f"{self.base_url}/document-store/refresh/{store_id}"
        payload = {"items": [item.to_dict() for item in items]}
        response = requests.post(url, json=payload, headers=self._get_headers())
        data = self._handle_response(response)
        if isinstance(data, list):
            return [UpsertResult.from_dict(item) for item in data]
        return [UpsertResult.from_dict(data)]

    def query_vector_store(self, query: RetrievalQuery) -> RetrievalResult:
        url = f"{self.base_url}/document-store/vectorstore/query"
        response = requests.post(url, json=query.to_dict(), headers=self._get_headers())
        data = self._handle_response(response)
        return RetrievalResult.from_dict(data)

    def create_document_store(self, store: DocumentStore) -> DocumentStore:
        url = f"{self.base_url}/document-store/store"
        response = requests.post(url, json=store.to_dict(), headers=self._get_headers())
        data = self._handle_response(response)
        return DocumentStore.from_dict(data)

    def update_chunk(self, store_id: str, loader_id: str, chunk_id: str, chunk: DocumentChunk) -> ChunkResponse:
        # Для update_chunk нужно импортировать DocumentChunk
        # Добавим импорт в начале файла
        url = f"{self.base_url}/document-store/chunks/{store_id}/{loader_id}/{chunk_id}"
        response = requests.put(url, json=chunk.to_dict(), headers=self._get_headers())
        data = self._handle_response(response)
        return ChunkResponse.from_dict(data)

    def update_document_store(self, store_id: str, store: DocumentStore) -> DocumentStore:
        url = f"{self.base_url}/document-store/store/{store_id}"
        response = requests.put(url, json=store.to_dict(), headers=self._get_headers())
        data = self._handle_response(response)
        return DocumentStore.from_dict(data)

    def delete_document_store(self, store_id: str) -> None:
        url = f"{self.base_url}/document-store/store/{store_id}"
        response = requests.delete(url, headers=self._get_headers())
        self._handle_response(response)

    def delete_chunk(self, store_id: str, loader_id: str, chunk_id: str) -> None:
        url = f"{self.base_url}/document-store/chunks/{store_id}/{loader_id}/{chunk_id}"
        response = requests.delete(url, headers=self._get_headers())
        self._handle_response(response)

    def delete_loader(self, store_id: str, loader_id: str) -> None:
        url = f"{self.base_url}/document-store/loader/{store_id}/{loader_id}"
        response = requests.delete(url, headers=self._get_headers())
        self._handle_response(response)

    def delete_vector_store_data(self, store_id: str) -> None:
        url = f"{self.base_url}/document-store/vectorstore/{store_id}"
        response = requests.delete(url, headers=self._get_headers())
        self._handle_response(response)