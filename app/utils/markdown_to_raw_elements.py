# app/utils/markdown_to_raw_elements.py

from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from utils.schemas import RawElement, DocCategory


class MarkdownToRawElements:
    """
    Преобразует Markdown-строку в список RawElement,
    разбивая документ по заголовкам, таблицам, спискам и блокам кода.
    
    Каждый элемент получает:
    - content: очищенный текст / таблица / элементы списка
    - category: подходящий DocCategory (DOCUMENT_CHUNK, TABLE, LIST, CODE)
    - title: иерархический заголовок (breadcrumbs)
    - метаданные исходного документа (source, url и т.д.)
    """

    # Пороговое значение: если параграф длиннее, его можно будет разбить позже
    MAX_PARAGRAPH_LENGTH = 2000  # символов, примерное ограничение

    def __init__(self):
        # Скомпилируем часто используемые регулярки
        self._title_re = re.compile(r'^TITLE: (.+)$')
        self._heading_re = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self._table_sep_re = re.compile(r'^\s*\|?\s*[-:]+\s*\|')  # разделитель заголовка таблицы
        self._list_item_re = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)')
        self._code_block_re = re.compile(r'^```', re.MULTILINE)

    def convert(
        self,
        markdown_text: str,
        *,
        source: str,
        title: str = "",
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> List[RawElement]:
        """
        Основной метод преобразования.
        
        :param markdown_text: исходный Markdown
        :param source: имя файла или идентификатор источника
        :param title: общий заголовок документа (если не задан, берётся из первого H1)
        :param url: ссылка на источник (опционально)
        :param metadata: дополнительные метаданные (автор, даты и т.п.)
        :param created_at: дата создания документа
        :param updated_at: дата изменения документа
        :return: список RawElement
        """
        base_meta = metadata or {}
        lines = markdown_text.splitlines(keepends=True)  # сохраняем переносы для контекста

        elements = []
        # Стек для хранения текущей иерархии заголовков (breadcrumbs)
        heading_stack = []  # элементы: (level, heading_text)
        current_text_lines = []  # накопитель строк для блока DOCUMENT_CHUNK
        in_table = False
        in_code_block = False
        table_lines = []
        list_lines = []

        # Если документ начинается с H1, используем его как общий title
        if not title:
            for line in lines:
                m = self._title_re.match(line.lstrip())
                if m and m.group(1):
                    title = m.group(1).strip()
                    lines.remove(line)
                    break

        def flush_text_block():
            """Сохраняет накопленный текстовый блок как DOCUMENT."""
            nonlocal current_text_lines
            if current_text_lines:
                text = ''.join(current_text_lines).strip()
                if text:
                    elements.append(self._make_element(
                        content=text,
                        category=DocCategory.DOCUMENT,
                        title=self._build_title(heading_stack),
                        source=source,
                        url=url,
                        base_meta=base_meta,
                        created_at=created_at,
                        updated_at=updated_at,
                    ))
                current_text_lines = []

        def flush_table():
            """Сохраняет накопленные строки таблицы как TABLE."""
            nonlocal table_lines
            if table_lines:
                table_text = ''.join(table_lines).strip()
                if table_text:
                    elements.append(self._make_element(
                        content=table_text,
                        category=DocCategory.TABLE,
                        title=self._build_title(heading_stack),
                        source=source,
                        url=url,
                        base_meta=base_meta,
                        created_at=created_at,
                        updated_at=updated_at,
                    ))
                table_lines = []

        def flush_list():
            """Сохраняет накопленный список как LIST."""
            nonlocal list_lines
            if list_lines:
                list_text = ''.join(list_lines).strip()
                if list_text:
                    elements.append(self._make_element(
                        content=list_text,
                        category=DocCategory.LIST,
                        title=self._build_title(heading_stack),
                        source=source,
                        url=url,
                        base_meta=base_meta,
                        created_at=created_at,
                        updated_at=updated_at,
                    ))
                list_lines = []

        def update_heading_stack(line: str):
            """Обрабатывает строку заголовка, обновляет heading_stack."""
            nonlocal heading_stack
            m = self._heading_re.match(line.lstrip())
            if not m:
                return False
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            # Убираем из стека все заголовки с уровнем >= текущего
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            return True

        def is_table_line(line: str) -> bool:
            """Проверяет, является ли строка частью markdown-таблицы."""
            return '|' in line

        def is_table_separator(line: str) -> bool:
            """Проверяет, является ли строка разделителем заголовка таблицы."""
            return bool(self._table_sep_re.match(line.strip()))

        i = 0
        while i < len(lines):
            raw_line = lines[i]
            stripped = raw_line.strip()

            # Блок кода
            if self._code_block_re.match(stripped):
                if in_code_block:
                    in_code_block = False
                    # закрыли блок – сохраним как CODE
                    flush_text_block()
                    # собираем код: от открывающей до закрывающей строки
                    code_lines = []
                    # проходим от предыдущего ``` до текущего
                    # (упрощение: будем считать, что код идёт целиком)
                    i += 1
                    continue  # в реальном коде нужно отдельно обрабатывать
                else:
                    in_code_block = True
                    flush_text_block()
                    flush_table()
                    flush_list()
                    # Начинаем сбор кода
                    j = i + 1
                    while j < len(lines) and not self._code_block_re.match(lines[j].strip()):
                        j += 1
                    code_content = ''.join(lines[i:j+1])  # включая закрывающий ```
                    elements.append(self._make_element(
                        content=code_content.strip(),
                        category=DocCategory.DOCUMENT_CHUNK,  # или ввести CODE, но пока так
                        title=self._build_title(heading_stack),
                        source=source,
                        url=url,
                        base_meta=base_meta,
                        created_at=created_at,
                        updated_at=updated_at,
                    ))
                    i = j + 1
                    continue

            # Обработка таблиц вне кодовых блоков
            if not in_code_block:
                if is_table_line(raw_line) and not is_table_separator(raw_line):
                    if not in_table:
                        flush_text_block()
                        flush_list()
                        in_table = True
                    table_lines.append(raw_line)
                    i += 1
                    continue
                elif in_table:
                    # Проверяем, закончилась ли таблица
                    if not is_table_line(raw_line) or is_table_separator(raw_line):
                        flush_table()
                        in_table = False
                        # не увеличиваем i, чтобы повторно обработать текущую строку
                        continue
                    else:
                        table_lines.append(raw_line)
                        i += 1
                        continue

                # Обработка списков
                list_match = self._list_item_re.match(raw_line)
                if list_match and not in_table:
                    if not list_lines:
                        flush_text_block()
                    list_lines.append(raw_line)
                    i += 1
                    continue
                elif list_lines:
                    flush_list()
                    continue  # перепроверить текущую строку

            # Обработка заголовков
            if not in_table and not in_code_block and not list_lines:
                if update_heading_stack(raw_line):
                    flush_text_block()
                    i += 1
                    continue

            # Обычный текст
            if not in_table and not in_code_block and not list_lines:
                current_text_lines.append(raw_line)
            i += 1

        # Финализация остатков
        flush_table()
        flush_list()
        flush_text_block()

        # Если заголовок документа так и не был установлен, используем source
        if not title:
            title = source

        # Проставляем общий title первого уровня всем элементам, у которых title пуст
        for elem in elements:
            if not elem.title:
                elem.title = title
        return elements

    def _build_title(self, heading_stack: List[tuple]) -> str:
        """Собирает breadcrumbs из стека заголовков."""
        if not heading_stack:
            return ""
        return " > ".join(level_text[1] for level_text in heading_stack)

    def _make_element(
        self,
        content: str,
        category: DocCategory,
        title: str,
        source: str,
        url: Optional[str],
        base_meta: Dict[str, Any],
        created_at: Optional[datetime],
        updated_at: Optional[datetime],
    ) -> RawElement:
        """Создаёт RawElement с обогащёнными метаданными."""
        return RawElement(
            content=content,
            source=source,
            category=category,
            title=title,
            url=url,
            metadata={
                **base_meta,
                "converter": "MarkdownToRawElements",
                "heading_path": title,
            },
            created_at=created_at,
            updated_at=updated_at,
        )