# app/merger/chunk_merger.py

from know_everything_ai.schemas import DocCategory, RawElement
from know_everything_ai.settings import Settings
from know_everything_ai.utils.token_counter import TokenEstimator


class ChunkMerger:
    """
    Склеивает слишком маленькие чанки одного типа в более крупные,
    чтобы улучшить качество эмбеддингов.
    """

    def __init__(
        self,
        min_tokens: int = 100,
        max_tokens: int = 800,
        estimator: TokenEstimator | None = None,
    ):
        """
        :param min_tokens: чанки короче этого порога будут объединяться
        :param max_tokens: итоговый чанк не превысит этот размер
        :param estimator: счётчик токенов; при None берётся из настроек по
            умолчанию, то есть без учёта сконфигурированного бэкенда
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.estimator = estimator or TokenEstimator(Settings())

    def count(self, text: str) -> int:
        return self.estimator.estimate(text)

    def merge(self, chunks: list[RawElement]) -> list[RawElement]:
        if not chunks:
            return []

        merged: list[RawElement] = []
        buffer: list[RawElement] = []   # накапливаемые чанки одного типа
        buffer_tokens = 0
        current_type = None
        current_title = None

        for chunk in chunks:
            # Игнорируем KNOWLEDGE_CANVAS и прочие особые типы – они не склеиваются
            if chunk.category in (DocCategory.KNOWLEDGE_CANVAS,):
                self._flush_buffer(buffer, merged, buffer_tokens, current_type, current_title)
                merged.append(chunk)
                continue

            # FAQ и глоссарий обычно уже хорошего размера, склеиваем только если они маленькие
            # но лучше их не трогать, чтобы не смешивать пары
            if chunk.category in (DocCategory.FAQ, DocCategory.GLOSSARY):
                self._flush_buffer(buffer, merged, buffer_tokens, current_type, current_title)
                merged.append(chunk)
                continue

            # Определяем тип и заголовок
            ctype = chunk.category
            ctitle = chunk.title  # заголовок (breadcrumbs)

            # Если тип или заголовок сменились – закрываем буфер
            if buffer and (ctype != current_type or ctitle != current_title):
                self._flush_buffer(buffer, merged, buffer_tokens, current_type, current_title)
                buffer = []
                buffer_tokens = 0

            # Добавляем чанк в буфер
            chunk_tokens = self.count(chunk.content)
            buffer.append(chunk)
            buffer_tokens += chunk_tokens
            current_type = ctype
            current_title = ctitle

            # Если буфер уже достаточно велик – сохраняем его
            if buffer_tokens >= self.max_tokens:
                self._flush_buffer(buffer, merged, buffer_tokens, current_type, current_title)
                buffer = []
                buffer_tokens = 0

        # Не забываем остатки
        self._flush_buffer(buffer, merged, buffer_tokens, current_type, current_title)
        return merged

    def _flush_buffer(self, buffer, merged, tokens, ctype, title):
        if not buffer:
            return
        if tokens < self.min_tokens and len(buffer) == 1:
            # Единственный короткий чанк – оставляем как есть
            merged.append(buffer[0])
        else:
            # Склеиваем содержимое
            combined_content = "\n\n".join(b.content for b in buffer)
            # Берём метаданные первого чанка (можно обогатить)
            base_meta = buffer[0].metadata.copy()
            base_meta["merged_from"] = [b.source for b in buffer]
            merged_chunk = RawElement(
                content=combined_content,
                source=buffer[0].source,
                category=ctype,
                title=title or buffer[0].title,
                url=buffer[0].url,
                metadata=base_meta,
                created_at=buffer[0].created_at,
                updated_at=buffer[0].updated_at,
            )
            merged.append(merged_chunk)
        buffer.clear()
