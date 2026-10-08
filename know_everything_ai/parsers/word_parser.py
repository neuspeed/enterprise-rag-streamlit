
import asyncio
import zipfile
from datetime import datetime
from pathlib import Path

import structlog
from bs4 import BeautifulSoup

from know_everything_ai.schemas import DocCategory, RawElement

log = structlog.get_logger("word_parser")

class WordParser:
    async def parse(self, file_path: Path) -> list[RawElement]:
        """Parse a .docx file into RawElement objects.

        The ZIP/XML work is synchronous and CPU-bound, so it runs in a worker
        thread: on the event loop it stalled every other concurrent job.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if file_path.suffix.lower() != ".docx":
            raise ValueError(
                f"Поддерживаются только .docx файлы, получено: {file_path.suffix!r}"
            )

        log.debug("word_parsing_start", file_path=str(file_path))
        try:
            elements = await asyncio.to_thread(self.__parse_docx, file_path)
        finally:
            file_path.unlink(missing_ok=True)
        log.debug("word_parsing_done", file_path=str(file_path), elements=len(elements))
        return elements

    def __parse_paragraph(self, paragraph, relationships) -> RawElement:
        # Обработка всех параграфов
        para_parts = []
        current_link = None
        link_parts = []

        for run in paragraph.find_all('w:r'):
            # Извлекаем текст
            text_elem = run.find('w:t')
            if text_elem and text_elem.string:
                text_content = text_elem.string

                # Проверяем, находится ли текст внутри гиперссылки
                hyperlink = run.find_parent('w:hyperlink')
                if hyperlink and hyperlink.get('r:id'):
                    link_id = hyperlink['r:id']
                    if link_id in relationships:
                        # Начало или продолжение ссылки
                        if current_link != link_id:
                            # Завершаем предыдущую ссылку, если была
                            if current_link:
                                url = relationships[current_link]
                                para_parts.append(f"{''.join(link_parts)} ({url})")
                                link_parts = []
                            current_link = link_id

                        link_parts.append(text_content)
                        continue

                # Если есть накопленный текст ссылки, добавляем его
                if current_link:
                    url = relationships[current_link]
                    para_parts.append(f"{''.join(link_parts)} ({url})")
                    current_link = None
                    link_parts = []

                # Добавляем обычный текст
                para_parts.append(text_content)

        # Добавляем оставшуюся ссылку, если есть
        if current_link:
            url = relationships[current_link]
            para_parts.append(f"{''.join(link_parts)} ({url})")

        # Объединяем части параграфа
        return RawElement(
                        content=''.join(para_parts),
                        category=DocCategory.DOCUMENT,
                    )

    def __parse_table(self, table, relationships) -> RawElement:
        result_text = []
        # Обработка таблиц
        for row in table.find_all('w:tr'):
            row_parts = []
            for cell in row.find_all('w:tc'):
                cell_parts = []
                current_link = None
                link_parts = []

                for run in cell.find_all('w:r'):
                    # Извлекаем текст
                    text_elem = run.find('w:t')
                    if text_elem and text_elem.string:
                        text_content = text_elem.string

                        # Проверяем гиперссылку
                        hyperlink = run.find_parent('w:hyperlink')
                        if hyperlink and hyperlink.get('r:id'):
                            link_id = hyperlink['r:id']
                            if link_id in relationships:
                                # Начало или продолжение ссылки
                                if current_link != link_id:
                                    # Завершаем предыдущую ссылку
                                    if current_link:
                                        url = relationships[current_link]
                                        cell_parts.append(f"{''.join(link_parts)} ({url})")
                                        link_parts = []
                                    current_link = link_id

                                link_parts.append(text_content)
                                continue

                        # Если есть накопленный текст ссылки
                        if current_link:
                            url = relationships[current_link]
                            cell_parts.append(f"{''.join(link_parts)} ({url})")
                            current_link = None
                            link_parts = []

                        # Добавляем обычный текст
                        cell_parts.append(text_content)

                # Добавляем оставшуюся ссылку
                if current_link:
                    url = relationships[current_link]
                    cell_parts.append(f"{''.join(link_parts)} ({url})")

                # Объединяем содержимое ячейки
                row_parts.append(''.join(cell_parts).strip())

            # Объединяем ячейки строки
            result_text.append(' | '.join(row_parts))
        return RawElement(
                        content='\n'.join(result_text).strip(),
                        category=DocCategory.TABLE
                    )


    def __parse_docx(self, file_path) -> list[RawElement]:
        """Надежный парсер .docx файлов с сохранением ссылок и таблиц"""
        with zipfile.ZipFile(file_path) as z:
            with z.open('word/_rels/document.xml.rels') as f:
                rels_content = f.read()

            with z.open('word/document.xml') as f:
                doc_content = f.read()

        rels_soup = BeautifulSoup(rels_content, 'xml')
        doc_soup = BeautifulSoup(doc_content, 'xml')

        relationships = {}
        for rel in rels_soup.find_all('Relationship'):
            if rel.get('Id') and rel.get('Target'):
                relationships[rel['Id']] = rel['Target']

        modified = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
        elements = []

        for elem in doc_soup.find_all(['w:p', 'w:tbl']):
            if elem.name == 'tbl':
                element = self.__parse_table(elem, relationships)
            elif elem.find_parent('w:tbl') is None:
                # Paragraphs inside a table are already covered by the table
                # element. Skipping them here matters: previously they fell
                # through both branches and re-appended the previously parsed
                # element, duplicating content once per nested paragraph.
                element = self.__parse_paragraph(elem, relationships)
            else:
                continue

            if element.content.strip():
                element.created_at = modified
                element.updated_at = modified
                element.source = Path(file_path).name
                elements.append(element)

        return elements

