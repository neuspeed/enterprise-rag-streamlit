import asyncio
import aiohttp
import structlog
import json
from typing import Dict, List, Any
from tempfile import NamedTemporaryFile

from cleaners.text_cleaner import clean_text
from utils.markdown_to_raw_elements import MarkdownToRawElements
from classifier.chunk_classifier import ChunkClassifier
from structurizers.faq_structurizer import FAQStructurizer
from structurizers.glossary_structurizer import GlossaryStructurizer
from enricher.knowledge_canvas import KnowledgeCanvasEnricher
from loaders.registry import LOADER_EXTENSION_MAP
from utils.token_counter import estimate_tokens
from flowise.client import Flowise
from flowise.models import (
    UpsertConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RecordManagerConfig,
    BaseConfig,
    UpsertResult,
    DocumentStore
)
from utils.webhook_utils import WebhookSender
from utils.schemas import RawElement, DocCategory, Payload, ReturnPayload
from utils.chunk_merger import ChunkMerger
from settings import Settings


log = structlog.get_logger("knowledge_pipeline")

class KnowledgePipeline:
    KB_NAME_PREFIX = "kb_"
    
    def __init__(self, settings = None):
        self.settings: Settings = settings
        self.enricher = KnowledgeCanvasEnricher(
            api_url=self.settings.ENRICHER_API_URL, 
            api_key=self.settings.ENRICHER_API_KEY,
            model=self.settings.ENRICHER_MODEL_NAME,
            system_prompt=self.settings.ENRICHER_PROMPT
        )
        self.classifier = ChunkClassifier(
            api_url=self.settings.CLASSIFIER_API_URL, 
            api_key=self.settings.CLASSIFIER_API_KEY,
            model=self.settings.CLASSIFIER_MODEL_NAME,
            system_prompt=self.settings.CLASSIFIER_PROMPT_TEMPLATE,
            category_descriptions=self.settings.CLASSIFIER_CATEGORY_DESCRIPTIONS
        )
        self.faq_struct = FAQStructurizer(self.settings.LLM_STRUCT)
        self.glossary_struct = GlossaryStructurizer(self.settings.LLM_STRUCT)
        self.splitter = MarkdownToRawElements()
        self.flowise = Flowise(self.settings.FLOWISE_API_URL, self.settings.FLOWISE_API_KEY)
        self.embedding = EmbeddingConfig.huggingface_inference(
            model=settings.EMBEDDING_MODEL,
            credential=settings.EMBEDDING_CREDENTIAL,
            api_url=settings.EMBEDDING_API_URL
        )
        self.chunk_merger = ChunkMerger()
        self.max_enrich_tokens = self.settings.MAX_CKB_TOKENS 
        self.webhook_sender = WebhookSender(self.settings)

    async def run(self, payload: Payload) -> int:
        """
        Главный метод обработки одного сообщения.
        Возвращает количество загруженных в Flowise элементов.
        """
        # 1. Скачиваем и загружаем все файлы -> список RawElement (по одному на документ)
        log.debug("documents_loading_start")
        raw_documents = await self._load_all_documents(payload)
        if not raw_documents:
            raise FileNotFoundError(payload.data)
        merged_chunks = self.chunk_merger.merge(raw_documents)
        log.debug("documents_loading_done")
        if not merged_chunks:
            raise ValueError(payload.data)

        # 2. Очистка и оценка суммарного размера
        log.debug("documents_cleaning_start")
        for doc in merged_chunks:
            doc.content = clean_text(doc.content)
        log.debug("documents_cleaning_done")
        log.debug("token_counting_start")
        total_tokens = sum(estimate_tokens(doc.content) for doc in merged_chunks)
        log.debug("token_counting_done", total_tokens=total_tokens)
        # 3. Ветвление
        kb_external_id = payload.kb_external_id
        if not kb_external_id:
            raise ValueError("kb_external_id is required")
        ds: DocumentStore = await self._get_or_create_document_store(kb_external_id)
        if payload.kb_type == "context":
            if total_tokens > self.max_enrich_tokens:
                raise ValueError(
                    f"Cannot create context KB: total tokens {total_tokens} exceeds limit {self.max_enrich_tokens}"
                )
            log.debug("processing_as_ckb_start")
            output_items = await self._process_as_small(merged_chunks)
            if not output_items:
                log.info("pipeline", message="All data classified as not useful")
                return_payload = ReturnPayload(
                    id=payload.id,
                    job_id=payload.job_id,
                    text_out="",
                    cost={},
                    status="error",
                    reason="All data classified as not useful",
                    kb_type="context"
                )
                self.webhook_sender.send_to_webhook(
                    external_url=payload.external_url,
                    data=return_payload.model_dump_json()
                
                )
                return
            return_payload = ReturnPayload(
                id = payload.id,
                job_id= payload.job_id,
                text_out=json.dumps(output_items),
                cost={},
                kb_type="context",
                status="success"
            )
            self.webhook_sender.send_to_webhook(
                external_url=payload.external_url,
                data=return_payload.model_dump_json()
            )
            log.debug("processing_as_ckb_done", output_items=output_items) 
        elif payload.kb_type == "vector":
            log.debug("processing_as_vkb_start")
            output_items = await self._process_as_large(merged_chunks)
            if not output_items:
                log.info("pipeline", message="All data classified as not useful")
                return_payload = ReturnPayload(
                    id=payload.id,
                    job_id=payload.job_id,
                    text_out="",
                    cost={},
                    status="error",
                    reason="All data classified as not useful",
                    kb_type="vector"
                )
                self.webhook_sender.send_to_webhook(
                    external_url=payload.external_url,
                    data=return_payload.model_dump_json()
                
                )
                return
            # 4. Отправка в Flowise
            log.debug("flowise_document_upsertion_start", output_items=output_items)
            self._flowise_upsert_documents(output_items, ds=ds)
            return_payload = ReturnPayload(
                id = payload.id,
                job_id= payload.job_id,
                text_out=ds.id,
                cost={},
                kb_type="vector",
                status="success"
            )
            self.webhook_sender.send_to_webhook(
                external_url=payload.external_url,
                data=return_payload.model_dump_json()
            )
            log.debug("flowise_document_upsertion_done", ds_id=ds.id)
            log.debug("processing_as_vkb_done")
        else:
            if total_tokens <= self.max_enrich_tokens:
                log.debug("processing_as_ckb_start")
                output_items = await self._process_as_small(merged_chunks)
                if not output_items:
                    log.info("pipeline", message="All data classified as not useful")
                    return_payload = ReturnPayload(
                        id=payload.id,
                        job_id=payload.job_id,
                        text_out="",
                        cost={},
                        status="error",
                        reason="All data classified as not useful",
                        kb_type="context"
                    )
                    self.webhook_sender.send_to_webhook(
                        external_url=payload.external_url,
                        data=return_payload.model_dump_json()
                    
                    )
                    return
                return_payload = ReturnPayload(
                    id = payload.id,
                    job_id= payload.job_id,
                    text_out=json.dumps(output_items),
                    cost={},
                    kb_type="context",
                    status="success"
                )
                self.webhook_sender.send_to_webhook(
                    external_url=payload.external_url,
                    data=return_payload.model_dump_json()
                )
                log.debug("processing_as_ckb_done", output_items=output_items)  
            else:
                log.debug("processing_as_vkb_start")
                output_items = await self._process_as_large(merged_chunks)
                if not output_items:
                    log.info("pipeline", message="All data classified as not useful")
                    return_payload = ReturnPayload(
                        id=payload.id,
                        job_id=payload.job_id,
                        text_out="",
                        cost={},
                        status="error",
                        reason="All data classified as not useful",
                        kb_type="vector"
                    )
                    self.webhook_sender.send_to_webhook(
                        external_url=payload.external_url,
                        data=return_payload.model_dump_json()
                    
                    )
                    return
                # 4. Отправка в Flowise
                log.debug("flowise_document_upsertion_start", output_items=output_items)
                self._flowise_upsert_documents(output_items, ds=ds)
                return_payload = ReturnPayload(
                    id = payload.id,
                    job_id= payload.job_id,
                    text_out=ds.id,
                    cost={},
                    kb_type="vector",
                    status="success"
                )
                self.webhook_sender.send_to_webhook(
                    external_url=payload.external_url,
                    data=return_payload.model_dump_json()
                )
                log.debug("flowise_document_upsertion_done", ds_id=ds.id)
                log.debug("processing_as_vkb_done")
        return len(output_items)
    


    # ------------------------------------------------------------------
    # Загрузка документов
    # ------------------------------------------------------------------
    async def _load_all_documents(self, payload: Payload) -> List[RawElement]:
        all_docs = []
        if isinstance(payload.data, str):
            try:
                data = json.loads(payload.data)
            except json.JSONDecodeError as e:
                log.error("load_document_error", error=f"Failed to parse JSON from payload.data: {e}")
                raise
        else:
            data = payload.data
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Файлы из поля "files" (docx, pdf, csv, xlsx...)
            if data.get("files"):
                for file_type, urls in data.get("files", {}).items():
                    loader = LOADER_EXTENSION_MAP[file_type]
                    for url in urls:
                        tasks.append(loader.transform(url))

            # HTML из поля "html"
            html_loader = LOADER_EXTENSION_MAP["html"]
            for url in data.get("html", []):
                tasks.append(html_loader.transform(url))

            # Простой текст
            if "text" in data and data["text"]:
                all_docs.append(RawElement(
                    content=data["text"],
                    source="inline_text",
                    category=DocCategory.DOCUMENT,
                    title="",
                    metadata={}
                ))

            results = await asyncio.gather(*tasks, return_exceptions=False)
            for res in results:
                if isinstance(res, Exception):
                    log.error(f"Load error: {res}")
                else:
                    all_docs.extend(res)  # список RawElement
        return all_docs


    # ------------------------------------------------------------------
    # Ветка малого объёма – Enricher
    # ------------------------------------------------------------------
    async def _process_as_small(self, raw_docs: List[RawElement]) -> List[dict]:
        """Склеивает все документы и создаёт единое knowledge canvas."""
        combined_text = "\n\n".join(
            f"# Источник: {doc.source}\n{doc.content}" for doc in raw_docs
        )
        enriched = await self.enricher.create_knowledge_canvas(combined_text, {})
        
        # enriched – строка или словарь, мы упаковываем в один элемент
        return [{
            "content": enriched,
            "category": DocCategory.KNOWLEDGE_CANVAS.value,
            "source": "merged_documents",
            "created_at": None,
            "updated_at": None,
            "url": None,
            "metadata": {
                "original_documents":  list(dict.fromkeys(doc.source for doc in raw_docs)),
                "total_original_tokens": sum(estimate_tokens(d.content) for d in raw_docs)
            }
        }]

    # ------------------------------------------------------------------
    # Ветка большого объёма – Split → Classify → Structure
    # ------------------------------------------------------------------
    async def _process_as_large(self, raw_docs: List[RawElement]) -> List[dict]:
        """Каждый документ разбивается, классифицируется и структурируется."""
        all_items = []
        for doc in raw_docs:
            # 1. Split: преобразуем Markdown в структурные блоки
            structural_chunks = self.splitter.convert(
                doc.content,
                source=doc.source,
                title=doc.title,
                url=doc.url,
                metadata=doc.metadata,
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )

            for chunk in structural_chunks:
                # 2. Classify: уточняем категорию и полезность
                classified = await self.classifier.classify(chunk)
                log.debug("classifier_result", classified=classified, chunk_content=chunk.content)
                if not classified.is_useful:
                    continue

                # 3. Structure: извлекаем FAQ/глоссарий/оставляем как есть
                if classified.category == DocCategory.FAQ:
                    items = await self.faq_struct.extract(chunk)
                elif classified.category == DocCategory.GLOSSARY:
                    items = await self.glossary_struct.extract(chunk)
                else:
                    # chunk уже содержит category от MarkdownSplitter (TABLE, LIST, DOCUMENT_CHUNK)
                    items = [{
                        "content": chunk.content,
                        "category": classified.category.value,
                        "metadata": chunk.metadata,
                        "source": chunk.source,
                        "title": chunk.title,
                    }]
                all_items.extend(items)
    
        return all_items
    
       
    async def _get_or_create_document_store(self, kb_external_id: str) -> DocumentStore:
        """
        Возвращает ID Document Store, создавая его при необходимости.
        Имя DS = kb_<external_id>.
        """
        ds_name = f"{self.KB_NAME_PREFIX}{kb_external_id}"
        # Пытаемся найти существующий
        existing = self.flowise.document_store.find_document_store_by_name(ds_name)
        if existing:
            return existing
        # Создаём новый
        new_ds = self.flowise.document_store.create_document_store(store=DocumentStore(name=ds_name))
        return new_ds
    
    def _flowise_upsert_documents(self, items: List[Dict], ds: DocumentStore) -> UpsertResult:
        table_name = f"{ds.name.lower().replace('-','_')}_vec_table" 
        vector_store = VectorStoreConfig.postgres(
            host=self.settings.VECTOR_STORE_HOST,
            port=self.settings.VECTOR_STORE_PORT,
            database=self.settings.VECTOR_STORE_DATABASE,
            table=table_name,
            topK=10,
            credential=self.settings.VECTOR_STORE_CREDENTIAL
        )
        record_manager = RecordManagerConfig.postgres(
            host=self.settings.RECORD_MANAGER_HOST,
            port=self.settings.RECORD_MANAGER_PORT,
            database=self.settings.RECORD_MANAGER_DATABASE,
            table=f"{table_name}_upsertion_records",
            credential=self.settings.VECTOR_STORE_CREDENTIAL
        )
        
        loader = BaseConfig(
            "jsonFile",
            {
                "metadata": self.__keys_to_json(items),
                "separateByObject": True,
            }
        )

        config = UpsertConfig(
            loader=json.dumps(loader.to_dict()),
            embedding=json.dumps(self.embedding.to_dict()),
            vectorStore=json.dumps(vector_store.to_dict()),
            recordManager=json.dumps(record_manager.to_dict()),
            loaderName="test",
            docStore=json.dumps({"name":ds.name})
        )
        
        files = { "files": ( "test.json", json.dumps(items).encode(), "application/json")}
        result = self.flowise.document_store.upsert_document(store_id=ds.id, config=config, files=files)
        return result
    
    
    def __keys_to_json(self, list_of_dicts: List[Dict[str, Any]]) -> str:
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
    


async def main():
    payload = Payload(
        id = 101,
        job_id= 1023,
        kb_type="vector",
        kb_external_id="client123_project_abc", 
        data={
            "docx": [
            "https://cloud.infercom.one/remote.php/webdav/test/knowledge_pipeline/test_kolobok.docx",
            ],
        },
        external_url="https://webhook.site/f1f3e740-6e86-4a11-a378-5a4b0fa9f4a9"
    )
    pipeline = KnowledgePipeline(settings=Settings())
    await pipeline.run(payload)
    

if __name__ == "__main__":
    # ---------------------------  SETTINGS  ---------------------------
    # Максимальное время работы всего скрипта (сек.)
    GLOBAL_TIMEOUT = 900          # ← поменяйте под свои нужды
    # -----------------------------------------------------------------

    try:
        asyncio.run(main())
        # asyncio.wait_for бросит asyncio.TimeoutError, если время вышло
        #asyncio.run(asyncio.wait_for(main(), timeout=GLOBAL_TIMEOUT))
    except asyncio.TimeoutError:
        log.error(f"❌ Pipeline превысил глобальный таймаут {GLOBAL_TIMEOUT}s")
    except Exception as exc:                     # любые остальные ошибки
        log.exception(f"❌ Ошибка во время выполнения pipeline: {exc}")