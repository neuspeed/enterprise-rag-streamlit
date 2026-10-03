"""hybrid_chunker.py – замена LLM-чанкеру на локальные ML/правила."""
import re
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Попытка загрузить spaCy, иначе fallback на регулярки
try:
    import spacy
    nlp = spacy.load("ru_core_news_sm")
    nlp.add_pipe("sentencizer")  # быстрый токенизатор предложений
    USE_SPACY = True
except (ImportError, OSError):
    USE_SPACY = False
    logger.warning("spaCy ru_core_news_sm не найден, использую regex-разбиение.")


class ChunkConfig:
    """Параметры чанкинга."""
    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 500,
        hard_max_chunk_size: int = 3000,
        table_rows_per_chunk: int = 20,
        qa_max_size: int = 2500
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.hard_max_chunk_size = hard_max_chunk_size  # после склейки коротких не превышаем это
        self.table_rows_per_chunk = table_rows_per_chunk
        self.qa_max_size = qa_max_size


def _split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения. Использует spaCy или regex."""
    if USE_SPACY:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Простой regex: точка/вопрос/воскл. знак, за которым пробел и заглавная буква (или конец строки)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def _chunk_sentences(sentences: List[str], max_size: int) -> List[str]:
    """Группирует предложения в чанки не длиннее max_size."""
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            if current_chunk:
                current_chunk += " " + sent
            else:
                current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def _merge_short_chunks(chunks: List[Dict], min_size: int, hard_max: int) -> List[Dict]:
    """Склеивает соседние чанки одного source_file, если они короче min_size."""
    if not chunks:
        return chunks

    merged = []
    buffer = chunks[0].copy()
    for next_chunk in chunks[1:]:
        combined_len = len(buffer['content']) + len(next_chunk['content'])
        # Склеиваем, только если буфер короткий И после склейки не превысим hard_max
        if len(buffer['content']) < min_size and combined_len <= hard_max:
            buffer['content'] += "\n\n" + next_chunk['content']
        else:
            merged.append(buffer)
            buffer = next_chunk.copy()
    merged.append(buffer)
    return merged


def _assign_chunk_indices(chunks: List[Dict]) -> List[Dict]:
    """Присваивает chunk_index в порядке следования (глобально)."""
    for i, ch in enumerate(chunks):
        ch['chunk_index'] = i
    return chunks


def chunk_elements(raw_elements: List[Dict], config: Optional[ChunkConfig] = None) -> List[Dict]:
    """
    Главная функция чанкинга.
    Принимает список RawElement (словарей) и возвращает список чанков (словарей).
    """
    if config is None:
        config = ChunkConfig()

    # 1. Группировка элементов по source_file
    groups: Dict[str, List[Dict]] = {}
    for elem in raw_elements:
        source = elem.get('source_file', 'unknown')
        groups.setdefault(source, []).append(elem)

    all_chunks = []

    for source_file, items in groups.items():
        # Определяем категорию по первому элементу (предполагаем, что все в группе одной категории)
        category = items[0].get('category', 'ARTICLE')
        logger.debug(f"Обработка {source_file} [{category}], элементов: {len(items)}")

        if category == 'QA':
            # Каждый элемент — QA-пара
            for elem in items:
                content = elem['content']
                if len(content) > config.qa_max_size:
                    logger.warning(
                        f"QA-пара в {source_file} длинная ({len(content)} симв.), "
                        f"разрезаю по предложениям (потеря связности)."
                    )
                    sentences = _split_sentences(content)
                    sub_chunks = _chunk_sentences(sentences, config.max_chunk_size)
                    for sub in sub_chunks:
                        all_chunks.append({**elem, 'content': sub, 'category': category})
                else:
                    all_chunks.append({**elem, 'category': category})

        elif category == 'TABLE':
            # Группируем строки таблицы в чанки по table_rows_per_chunk строк
            # или по max_chunk_size (что наступит раньше)
            rows = [elem['content'] for elem in items]
            chunk_rows = []
            current_len = 0
            for row in rows:
                chunk_rows.append(row)
                current_len += len(row)
                if len(chunk_rows) >= config.table_rows_per_chunk or current_len >= config.max_chunk_size:
                    chunk_content = "\n".join(chunk_rows)
                    # Берём метаданные из первого элемента группы (они одинаковы)
                    meta = {k: items[0][k] for k in ['source_file', 'url', 'title']}
                    all_chunks.append({
                        **meta,
                        'content': chunk_content,
                        'category': category
                    })
                    chunk_rows = []
                    current_len = 0
            if chunk_rows:
                chunk_content = "\n".join(chunk_rows)
                meta = {k: items[0][k] for k in ['source_file', 'url', 'title']}
                all_chunks.append({
                    **meta,
                    'content': chunk_content,
                    'category': category
                })

        elif category == 'ARTICLE':
            # Статья: объединяем все абзацы в один текст и чанкаем по предложениям
            full_text = "\n\n".join([elem['content'] for elem in items])
            sentences = _split_sentences(full_text)
            raw_chunks = _chunk_sentences(sentences, config.max_chunk_size)
            meta = {k: items[0][k] for k in ['source_file', 'url', 'title']}
            for ch in raw_chunks:
                all_chunks.append({
                    **meta,
                    'content': ch,
                    'category': category
                })

        elif category == 'JSON_DATA':
            # Каждый элемент — самостоятельный JSON-объект
            for elem in items:
                all_chunks.append({**elem, 'category': category})

        else:
            # UNKNOWN или иное — обрабатываем как статью
            logger.warning(f"Неизвестная категория {category}, обрабатываю как ARTICLE.")
            full_text = "\n\n".join([elem['content'] for elem in items])
            sentences = _split_sentences(full_text)
            raw_chunks = _chunk_sentences(sentences, config.max_chunk_size)
            meta = {k: items[0][k] for k in ['source_file', 'url', 'title']}
            for ch in raw_chunks:
                all_chunks.append({
                    **meta,
                    'content': ch,
                    'category': 'UNKNOWN'
                })

    # 2. Постобработка: склейка коротких чанков в пределах одного source_file
    # Группируем чанки по source_file для склейки
    merged_by_file = []
    file_groups: Dict[str, List[Dict]] = {}
    for ch in all_chunks:
        file_groups.setdefault(ch['source_file'], []).append(ch)

    final_chunks = []
    for source, chunks in file_groups.items():
        merged = _merge_short_chunks(chunks, config.min_chunk_size, config.hard_max_chunk_size)
        final_chunks.extend(merged)

    # 3. Присваиваем индексы глобально (или можно в рамках source_file, но оставим глобально для упрощения)
    final_chunks = _assign_chunk_indices(final_chunks)

    logger.info(f"Всего чанков после гибридного чанкера: {len(final_chunks)}")
    return final_chunks


# ================== Интеграция в пайплайн ==================
if __name__ == "__main__":
    # Пример использования: читаем raw_elements.json, чанкаем, сохраняем chunks.json
    import json
    from pathlib import Path

    input_file = Path("./_before_semantic_chunking/raw_elements.json")
    output_file = Path("./_before_upsert/chunks.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_elements = json.load(f)

    chunks = chunk_elements(raw_elements)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Создано {len(chunks)} чанков, сохранено в {output_file}")