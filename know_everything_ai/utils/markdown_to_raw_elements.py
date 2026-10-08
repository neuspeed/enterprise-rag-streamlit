# app/utils/markdown_to_raw_elements.py

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from know_everything_ai.schemas import DocCategory, RawElement


class MarkdownToRawElements:
    """
    Преобразует Markdown-строку в список RawElement,
    разбивая документ по заголовкам, таблицам, спискам и блокам кода.

    Каждый элемент получает:
    - content: очищенный текст / таблица / элементы списка
    - category: подходящий DocCategory (DOCUMENT, TABLE, LIST, CODE)
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
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> list[RawElement]:
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
        heading_stack: list[tuple[int, str]] = []  # элементы: (level, heading_text)
        current_text_lines: list[str] = []  # накопитель строк для блока DOCUMENT
        in_table = False
        table_lines: list[str] = []
        list_lines: list[str] = []

        # Если документ начинается с H1, используем его как общий title
        if not title:
            for index, line in enumerate(lines):
                m = self._title_re.match(line.lstrip())
                if m and m.group(1):
                    title = m.group(1).strip()
                    del lines[index]
                    break

        def flush_text_block():
            """Сохраняет накопленный текстовый блок как DOCUMENT."""
            nonlocal current_text_lines
            if current_text_lines:
                text = ''.join(current_text_lines).strip()
                for piece in self._split_long_text(text):
                    elements.append(self._make_element(
                        content=piece,
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
            """Проверяет, является ли строка частью markdown-таблицы.

            Одной трубы недостаточно: иначе любая строка текста с символом «|»
            (например, описание пайпа в документации) начинает таблицу.
            """
            stripped = line.strip()
            return stripped.startswith('|') or stripped.count('|') >= 2

        def is_table_separator(line: str) -> bool:
            """Проверяет, является ли строка разделителем заголовка таблицы."""
            return bool(self._table_sep_re.match(line.strip()))

        i = 0
        while i < len(lines):
            raw_line = lines[i]
            stripped = raw_line.strip()

            # Fenced code block: collect from the opening fence to the closing
            # fence (or end of input) and emit it as one element. The previous
            # implementation referenced DocCategory.DOCUMENT_CHUNK, which does
            # not exist, so any document containing a code block raised
            # AttributeError and killed the whole ingestion job.
            if self._code_block_re.match(stripped):
                fence = stripped[:3]
                flush_text_block()
                flush_table()
                flush_list()

                end = i + 1
                while end < len(lines) and not lines[end].strip().startswith(fence):
                    end += 1

                last = min(end + 1, len(lines))
                elements.append(self._make_element(
                    content='\n'.join(lines[i:last]).strip(),
                    category=DocCategory.CODE,
                    title=self._build_title(heading_stack),
                    source=source,
                    url=url,
                    base_meta=base_meta,
                    created_at=created_at,
                    updated_at=updated_at,
                ))
                i = last
                continue

            # Обработка таблиц
            if is_table_line(raw_line) and not is_table_separator(raw_line):
                if not in_table:
                    flush_text_block()
                    flush_list()
                    in_table = True
                table_lines.append(raw_line)
                i += 1
                continue
            elif in_table:
                # Разделитель не прерывает таблицу: иначе заголовок уходит в
                # один элемент, а строки — в другой, и при эмбеддинге теряется
                # связь между ними.
                if is_table_separator(raw_line):
                    i += 1
                    continue
                if not is_table_line(raw_line):
                    flush_table()
                    in_table = False
                    # не увеличиваем i, чтобы повторно обработать текущую строку
                    continue
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
            if not in_table and not list_lines:
                if update_heading_stack(raw_line):
                    flush_text_block()
                    i += 1
                    continue

            # Обычный текст
            if not in_table and not list_lines:
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

    def _split_long_text(self, text: str) -> list[str]:
        """Break an oversized paragraph on blank lines.

        A document with no headings, tables or lists would otherwise become one
        element, which then fails embedding limits and skews the token estimate
        that selects the processing branch.
        """
        if len(text) <= self.MAX_PARAGRAPH_LENGTH:
            return [text] if text else []

        limit = self.MAX_PARAGRAPH_LENGTH
        blocks = re.split(r"\n\s*\n", text)
        pieces: list[str] = []
        buffer = ""

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if len(block) > limit:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                for start in range(0, len(block), limit):
                    pieces.append(block[start:start + limit])
                continue
            candidate = f"{buffer}\n\n{block}" if buffer else block
            if len(candidate) > limit:
                pieces.append(buffer)
                buffer = block
            else:
                buffer = candidate

        if buffer:
            pieces.append(buffer)
        return pieces

    def _build_title(self, heading_stack: list[tuple[int, str]]) -> str:
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
        url: str | None,
        base_meta: dict[str, Any],
        created_at: datetime | None,
        updated_at: datetime | None,
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
