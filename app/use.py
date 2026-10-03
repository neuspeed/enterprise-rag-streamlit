import uuid
import json
from typing import List,Dict,Any
from flowise import Flowise
from flowise.models import (
    UpsertConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RecordManagerConfig,
    BaseConfig
)

from settings import Settings



settings_local = Settings()

client = Flowise(
    base_url=settings_local.FLOWISE_API_URL,
    api_key=settings_local.FLOWISE_API_KEY)

# Используем HuggingFace Inference Embeddings
embedding = EmbeddingConfig.huggingface_inference(
    model=settings_local.EMBEDDING_MODEL,
    credential=settings_local.EMBEDDING_CREDENTIAL,
    api_url=settings_local.EMBEDDING_API_URL
)

vector_store = VectorStoreConfig.postgres(
    host=settings_local.VECTOR_STORE_HOST,
    port=settings_local.VECTOR_STORE_PORT,
    database=settings_local.VECTOR_STORE_DATABASE,
    table="test_flowise_upsert_table",
    topK=10,
    credential=settings_local.VECTOR_STORE_CREDENTIAL
)
record_manager = RecordManagerConfig.postgres(
    host=settings_local.RECORD_MANAGER_HOST,
    port=settings_local.RECORD_MANAGER_PORT,
    database=settings_local.RECORD_MANAGER_DATABASE,
    table="test_flowise_upsert_table_records",
    credential=settings_local.VECTOR_STORE_CREDENTIAL
)

def keys_to_json(list_of_dicts: List[Dict[str, Any]]) -> str:
    """
    Возвращает JSON, в котором каждому уникальному ключу (кроме 'content')
    из списка словарей сопоставлено значение "/<key>".

    Пример:
        >>> data = [
        ...     {"id": 1, "title": "Hello", "content": "text"},
        ...     {"id": 2, "author": "Bob", "content": "more text"},
        ... ]
        >>> print(keys_to_json(data))
        {"id": "/id", "title": "/title", "author": "/author"}

    :param list_of_dicts: список словарей произвольной структуры.
    :return: JSON‑строка вида {"key1": "/key1", "key2": "/key2", ...}
    """
    # 1️⃣ Собираем все ключи, кроме "content"
    unique_keys = set()
    for d in list_of_dicts:
        # Если элемент не словарь – просто пропускаем (можно изменить по желанию)
        if not isinstance(d, dict):
            continue
        for key in d.keys():
            if key != "content":
                unique_keys.add(key)

    # 2️⃣ Формируем словарь вида {key: f"/{key}"}
    result_dict = {key: f"/{key}" for key in sorted(unique_keys)}  # сортировка – для предсказуемого вывода

    # 3️⃣ Конвертируем в JSON
    return json.dumps(result_dict, ensure_ascii=False, indent=2)
with open("../test/test.json", 'rb') as f:
    content = f.read()
    
print(keys_to_json(json.loads(content.decode("utf-8"))))
loader = BaseConfig(
    "jsonFile",
    {
        "metadata": keys_to_json(json.loads(content.decode("utf-8"))),
        "separateByObject": True,
    }
)
import sys
#sys.exit("STOP")

config = UpsertConfig(
    loader=json.dumps(loader.to_dict()),
    embedding=json.dumps(embedding.to_dict()),
    vectorStore=json.dumps(vector_store.to_dict()),
    recordManager=json.dumps(record_manager.to_dict()),
    loaderName="test",
    docStore=json.dumps({"name":"AutoDocStore"})
)

files = { "files": ( "test.json", content, "application/json")}
result = client.document_store.upsert_document(store_id="daa56cd4-a59e-4de7-85b6-84bc41c0eaa7", config=config, files=files)
print(result.numAdded)