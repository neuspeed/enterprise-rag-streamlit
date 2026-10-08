#!/usr/bin/env python3

import asyncio
import html as _html_mod
from collections.abc import Callable, Mapping
from functools import partial

import chardet  # авто‑определение кодировки
import structlog
from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PageElement

from know_everything_ai.schemas import DocCategory, RawElement
from know_everything_ai.utils.markdown_to_raw_elements import MarkdownToRawElements

log = structlog.get_logger("html_parser")

#: Structural noise removed from every page. Site-specific selectors used to
#: live here; they made the parser silently drop content from any page that was
#: not the one they were written for.
NOISE_TAGS = ("script", "style", "head", "meta", "nav", "footer", "aside")


class HTMLParser:
    def __init__(self):
        self.DEFAULT_HANDLERS : dict[str, Callable[[Tag, list[str]], None]] = {   # Словарь со стандартными обработчиками
            "h1": self.__handle_title,
            "h2": self.__handle_heading,
            "h3": self.__handle_heading,
            "h4": self.__handle_heading,
            "h5": self.__handle_heading,
            "h6": self.__handle_heading,
            "br": self.__handle_br,
            "ul": self.__handle_list,
            "ol": self.__handle_list,
            "a": self.__handle_link,
            "table": self.__handle_table,
            "p": self.__handle_paragraph,
            "div": self.__handle_paragraph,
            "img": self.__handle_image,
            "title": self.__handle_title,
            # контейнеры таблицы – просто делегируем их дочерним элементам
            "thead": lambda t, o: None,
            "tbody": lambda t, o: None,
            "tfoot": lambda t, o: None,
        }

    async def parse(self, html_data: str, include_links: bool = False, decode_entities: bool = False, source: str = ""):
        """"Парсер для данных html"""
        if not html_data or not self._is_html(html_data):
            return [RawElement(content="Данных html не найдено", category=DocCategory.UNKNOWN)]
        log.debug("html_parsing_start", source=source)

        # BeautifulSoup is synchronous and CPU-bound; on the event loop it
        # blocked every other concurrent ingestion for the duration of the parse.
        result_content = await asyncio.to_thread(
            partial(
                self._html_to_text,
                html_data,
                include_links=include_links,
                decode_entities=decode_entities,
            )
        )

        if not result_content.strip():
            return [RawElement(content="Не удалось извлечь текст из HTML", category=DocCategory.UNKNOWN)]
        # The Markdown split walks the whole page with a dozen regexes and is
        # just as CPU-bound as the BeautifulSoup pass above it.
        elements = await asyncio.to_thread(
            MarkdownToRawElements().convert, result_content, source=source
        )
        log.debug("html_parsing_done", source=source, elements=len(elements))
        return elements

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    def _is_html(self, data: str) -> bool:
        """
        Простейшая проверка, что строка выглядит как HTML‑документ.
        Возвращает True, если в тексте найден один из «ключевых» маркеров:
            • <!DOCTYPE html>
            • <html ...>
            • <head ...>
            • <body ...>

        Тест делается без парсинга, т.е. работает быстро даже на больших файлах.
        """
        # Приводим к нижнему регистру, чтобы проверка была case‑insensitive.
        lowered = data.lower()

        # Поиск типичных маркеров HTML.
        markers = (
            "<!doctype html>",   # полное DOCTYPE
            "<html",             # открывающий тег <html …>
            "<head",             # открывающий тег <head …>
            "<body",             # открывающий тег <body …>
        )
        return any(m in lowered for m in markers)

    def _detect_encoding(self, data: bytes) -> str:
        """Определяем кодировку байтовой строки (fallback → utf‑8)."""
        result = chardet.detect(data)
        enc = result["encoding"]
        if enc and result["confidence"] > 0.5:
            return enc
        return "utf-8"


    def _clean_whitespace(self, text: str) -> str:
        """Удаляем лишние пробелы и оставляем один пустой разделитель."""
        lines = [line.strip() for line in text.splitlines()]
        cleaned = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned.append(line)
                prev_empty = False
            else:
                if not prev_empty:
                    cleaned.append("")
                    prev_empty = True
        return "\n".join(cleaned).strip()


    # ----------------------------------------------------------------------
    # Обработчики тегов (по умолчанию)
    # ----------------------------------------------------------------------
    def __handle_heading(self, tag: Tag, out: list[str]) -> None:
        """h1‑h6 → верхний регистр + пустая строка."""
        out.append(tag.get_text(separator=' ', strip=True).upper())
        out.append("")


    def __handle_br(self, _: Tag, out: list[str]) -> None:
        out.append("")


    def __handle_list(
            self,
            tag: Tag,
            out: list[str],
            handlers: Mapping[str, Callable[[Tag, list[str]], None]]) -> None:
        ordered = tag.name == "ol"
        for i, li in enumerate(tag.find_all("li", recursive=False), start=1):
            prefix = f"{i}." if ordered else "*"
            # Обрабатываем содержимое <li> рекурсивно, чтобы ссылки внутри списка тоже работали
            sub: list[str] = []
            self._recurse_children(li, sub, handlers)
            li_text = " ".join(sub).strip()
            out.append(f"{prefix} {li_text}")
        out.append("")


    def __handle_link(self, tag: Tag, out: list[str]) -> None:
        href = _attr(tag, "href")
        txt = tag.get_text(separator=' ', strip=True)
        if href:
            out.append(f"{txt} ({href})")
        else:
            out.append(txt)

    def __handle_link_without_url(self, tag: Tag, out: list[str]) -> None:
        txt = tag.get_text(separator=' ', strip=True)
        if txt:
            out.append(txt)

    def __handle_table(self, tag: Tag, out: list[str]) -> None:
        """Таблица → markdown‑подобный вывод."""
        # Сначала собираем все строки (включая <thead>, <tbody>, <tfoot>)
        rows: list[list[str]] = []
        for tr in tag.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                cells.append(cell.get_text(separator=' ', strip=True))
            rows.append(cells)

        if not rows:
            return

        # Находим максимальное количество колонок в таблице
        max_cols = max(len(row) for row in rows) if rows else 0

        # Выравниваем все строки до одинакового количества колонок
        aligned_rows = []
        for row in rows:
            aligned_row = row + [''] * (max_cols - len(row))  # Заполняем пустыми строками
            aligned_rows.append(aligned_row)

        # Вычисляем ширину колонок
        col_widths = [
            max(len(row[i]) for row in aligned_rows)
            for i in range(max_cols)
        ]

        def __format_row(row: list[str]) -> str:
            padded = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
            return "| " + " | ".join(padded) + " |"

        # Первая строка – заголовок
        out.append(__format_row(aligned_rows[0]))
        # Разделитель
        out.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
        # Оставшиеся строки
        for r in aligned_rows[1:]:
            out.append(__format_row(r))
        out.append("")   # пустая строка после таблицы


    def __handle_paragraph(self, _: Tag, out: list[str]) -> None:
        """Пустой обработчик – просто добавляем перевод строки после детей."""
        out.append("")   # будет заменено на один пустой разделитель в _clean_whitespace


    def __handle_image(self, tag: Tag, out: list[str]) -> None:
        alt = _attr(tag, "alt")
        src = _attr(tag, "src")
        if alt and src:
            out.append(f"[Изображение: {alt}] ({src})")
        elif alt:
            out.append(f"[Изображение: {alt}]")
        elif src:
            out.append(f"[Изображение] ({src})")
        else:
            out.append("[Изображение]")

    def __handle_title(self, tag: Tag, out: list[str]) -> None:
        title_text = tag.get_text(separator=' ', strip=True)
        if title_text:
            out.append(f"TITLE: {title_text}\n ...Content... ")
            out.append("")  # пустая строка после заголовка


    # ----------------------------------------------------------------------
    # Рекурсивный обход (внутри функции, но вынесен наружу для переиспользования)
    # ----------------------------------------------------------------------
    def _recurse_children(self, node: PageElement,
                        out: list[str],
                        handlers: Mapping[str, Callable[[Tag, list[str]], None]]) -> None:
        if isinstance(node, NavigableString):
            txt = str(node).strip()
            if txt:
                out.append(txt)
            return

        if not isinstance(node, Tag):
            return

        handler = handlers.get(node.name)
        if handler:
            if node.name in ("p", "div"):
                for child in node.children:
                    self._recurse_children(child, out, handlers)
                out.append("")
            else:
                if node.name in ("ul", "ol"):
                    self.__handle_list(node, out, handlers)
                else:
                    handler(node, out)
            return

        for child in node.children:
            self._recurse_children(child, out, handlers)


    # ----------------------------------------------------------------------
    # Основная функция конвертации
    # ----------------------------------------------------------------------
    def _html_to_text(
        self,
        html_content: str | bytes,
        *,
        tag_handlers: dict[str, Callable[[Tag, list[str]], None]] | None = None,
        include_links: bool = True,
        decode_entities: bool = True,
    ) -> str:
        """
        Преобразует HTML‑контент в чистый текст.

        Параметры
        ----------
        html_content : str | bytes
            Исходный HTML.
        tag_handlers : dict, optional
            Пользовательские обработчики тегов (переопределяют DEFAULT_HANDLERS).
        include_links : bool, default True
            Если False – ссылки выводятся без URL.
        decode_entities : bool, default True
            Декодировать HTML‑сущности.
        """
        # ── Приведение к строке ────────────────────────────────────────
        if isinstance(html_content, bytes):
            enc = self._detect_encoding(html_content)
            html_content = html_content.decode(enc, errors="replace")

        soup = BeautifulSoup(html_content, "lxml")

        head = soup.head                     # получаем <head>
        if head:                              # если он существует
            # Проходим только по непосредственным детям <head>
            for child in list(head.children):   # list() → копия, т.к. будем удалять
                # Комментарии, CDATA и переводы строк — NavigableString,
                # то есть str. Проверяем именно Tag, а не "не строка":
                # у PageElement нет атрибута name.
                if not isinstance(child, Tag):
                    continue
                # Оставляем только <title>
                if child.name != 'title':
                    child.decompose()


        # Убираем «шумные» элементы
        for elem in soup(NOISE_TAGS + ('[document]',)):
            elem.decompose()

        # Объединяем пользовательские обработчики
        handlers = dict(self.DEFAULT_HANDLERS)
        if tag_handlers:
            handlers.update(tag_handlers)

        # Отключаем вывод URL, если требуется
        if not include_links and "a" in handlers:
            handlers["a"] = self.__handle_link_without_url

        # ------------------------------------------------------------------
        # Обход дерева
        # ------------------------------------------------------------------
        out: list[str] = []

        root = soup.body if soup.body else soup
        self._recurse_children(root, out, handlers)

        raw = "\n".join(out)
        if not decode_entities:
        # «Экранируем» все символы, которые могли быть расшифрованы.
        # html.escape покрывает большинство проблемных символов.
            raw = _html_mod.escape(raw, quote=False)   # quote=False → не экранировать "
        else:
            raw = _html_mod.unescape(raw)

        return self._clean_whitespace(raw)


def _attr(tag: Tag, name: str) -> str:
    """Read an attribute as text.

    BeautifulSoup returns ``None`` when the attribute is absent and a list for
    multi-valued ones such as ``class`` and ``rel``, so calling ``.strip()``
    on the result directly raises on perfectly ordinary markup.
    """
    value = tag.get(name)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return " ".join(str(part) for part in value).strip()
