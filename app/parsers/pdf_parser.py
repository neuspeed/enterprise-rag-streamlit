# async_pdf_parser.py
import json
import base64
import structlog
import asyncio
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List

from pdf2image import convert_from_path

from utils.llm_utils import LLMDriven
from utils.schemas import RawElement, DocCategory
from settings import Settings

log = structlog.get_logger("pdf_parser")


class PDFParser(LLMDriven):
    """
    Асинхронный парсер PDF‑файлов.
    Внешний интерфейс – корутина `parse()`, которую можно вызывать из
    любого asyncio‑контекста (FastAPI, Celery‑worker с event‑loop,
    обычный скрипт с `asyncio.run()` и т.п.).
    """

    def __init__(self, api_url, api_key, model: str, system_prompt: str):
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt
        )
        self.settings = Settings()        
        # Ограничиваем количество одновременно обрабатываемых PDF,
        # чтобы не «переполнить» пул потоков и не отправить слишком
        # много запросов к VLM‑API.
        self._semaphore = asyncio.Semaphore(self.settings.MAX_CONCURRENT_PDFS or 4)

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #
    async def parse(self, file_path: Path) -> List[RawElement]:
        """Асинхронный входной метод."""
        async with self._semaphore:       
            return await self._parse(file_path)

    # --------------------------------------------------------------------- #
    # INTERNAL (async) IMPLEMENTATION
    # --------------------------------------------------------------------- #
    async def _parse(self, file_path: Path) -> List[RawElement]:
        """Обёртка, которая гарантирует удаление файла независимо от
        того, успел ли он обработаться или упал с исключением."""
        if not file_path.exists():
            return [
                RawElement(
                    content="Файл не найден",
                    category=DocCategory.UNKNOWN,
                )
            ]

        if file_path.suffix.lower() != ".pdf":
            return [
                RawElement(
                    content="Поддерживаются только .pdf файлы",
                    category=DocCategory.UNKNOWN,
                )
            ]

        try:
            log.debug("pdf_parsing_start", file_path=file_path)
            elements = await self.__parse_pdf(file_path)
            log.debug("pdf_parsing_done", file_path=file_path)
            return elements
        finally:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)

    async def __parse_pdf(self, file_path: Path) -> List[RawElement]:
        """
        Основная логика парсинга:
        1️⃣ Конвертируем PDF → список PIL‑изображений (в отдельном потоке)
        2️⃣ Кодируем каждое изображение в base64 (в отдельном потоке)
        3️⃣ Формируем payload и отправляем асинхронный запрос к VLM‑API
        4️⃣ Парсим ответ и формируем список `RawElement`
        """
        # ----------- 1. PDF → images ------------------------------------ #
        log.debug("convert_pdf_to_images_start", file_path=file_path)
        images: List[object] = await asyncio.to_thread(
            convert_from_path,
            file_path,
            dpi=200,
            poppler_path=self.settings.POPPLER_PATH
        )
        log.debug("convert_pdf_to_images_done", file_path=file_path, count=len(images))

        # ----------- 2. Формируем payload ------------------------------- #
        schema_json = RawElement.model_json_schema()
        prompt = self.system_prompt.format(
            schema=json.dumps(schema_json)
        )
        content = [
            {"type": "text", "text": prompt},
        ]

        # Кодируем изображения параллельно (группируем в gather)
        async def encode_one(img):
            return await asyncio.to_thread(self.__encode_image_to_base64, img)

        encoded_images = await asyncio.gather(*[encode_one(img) for img in images])

        for b64 in encoded_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        # ----------- 3. Асинхронный запрос к VLM ------------------------ #
        log.debug("vlm_pdf_processing_start", file_path=file_path)
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        log.debug("vlm_pdf_processing_done", file_path=file_path)

        # ----------- 4. Обрабатываем ответ ----------------------------- #
        result_content = (
            response.choices[0].message.content
            if getattr(response, "choices", None)
            else ""
        )
        if not result_content or not result_content.strip():
            return [
                RawElement(
                    content="Не удалось извлечь текст из PDF",
                    category=DocCategory.UNKNOWN,
                )
            ]

        try:
            chunks = json.loads(result_content)
            log.debug("pdf_parser", chunks=chunks)
            
            # Если пришёл объект, а не массив — оборачиваем
            if isinstance(chunks, dict):
                chunks = [chunks]
            # Если пришёл массив — оставляем как есть
            elif not isinstance(chunks, list):
                raise ValueError(f"Unexpected response type: {type(chunks)}")
                
        except json.JSONDecodeError as exc:
            # Попытка очистить ответ от markdown-разметки
            cleaned = result_content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            try:
                chunks = json.loads(cleaned)
                if isinstance(chunks, dict):
                    chunks = [chunks]
            except json.JSONDecodeError:
                return [
                    RawElement(
                        content=f"Ошибка разбора JSON‑ответа VLM: {exc}\nОтвет: {result_content[:200]}...",
                        category=DocCategory.UNKNOWN,
                    )
                ]

        # Приводим каждый chunk к схеме RawElement
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        elements = [
            RawElement(
                content=chunk.get("content", ""),
                category=DocCategory(chunk.get("category", DocCategory.UNKNOWN)),
                source=file_path.name,
                title=chunk.get("title", ""),
                metadata=chunk.get("metadata", {}),
                created_at=file_mtime,
                updated_at=file_mtime,
            )
            for chunk in chunks
        ]
        return elements

    # --------------------------------------------------------------------- #
    # Синхронный вспомогательный метод – вызывается через `to_thread`
    # --------------------------------------------------------------------- #
    @staticmethod
    def __encode_image_to_base64(pil_image) -> str:
        """Конвертирует PIL‑изображение в base64‑строку (JPEG, 85%‑ое качество)."""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
