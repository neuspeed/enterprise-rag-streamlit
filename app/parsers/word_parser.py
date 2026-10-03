
import traceback
import zipfile
import structlog
from datetime import datetime
from typing import List
from pathlib import Path
from bs4 import BeautifulSoup

from utils.schemas import RawElement, DocCategory

log = structlog.get_logger("word_parser")

class WordParser:
    async def parse(self, file_path: Path) -> List[RawElement]:   
        """Парсер для .docx файлов"""
        elements = []
        if not file_path.exists():
            return [RawElement(content="Файл не найден", category=DocCategory.UNKNOWN)]
        
        if not file_path.suffix.lower() == ".docx":
            return [RawElement(content="Поддерживаются только .docx файлы", category=DocCategory.UNKNOWN)]
            
        try:
            log.debug("word_parsing_start", file_path=file_path)
            elements = self.__parse_docx(file_path)
            log.debug("word_parsing_done", file_path=file_path)
        finally:
            file_path.unlink(missing_ok=True)
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
        

    def __parse_docx(self, file_path) -> List[RawElement]:
        """Надежный парсер .docx файлов с сохранением ссылок и таблиц"""
        try:
            # Открываем DOCX как ZIP-архив
            with zipfile.ZipFile(file_path) as z:
                # Читаем файл связей
                with z.open('word/_rels/document.xml.rels') as f:
                    rels_content = f.read()
                
                # Читаем основной документ
                with z.open('word/document.xml') as f:
                    doc_content = f.read()
            
            # Парсим XML
            rels_soup = BeautifulSoup(rels_content, 'xml')
            doc_soup = BeautifulSoup(doc_content, 'xml')
            
            # Создаем словарь связей ID -> URL
            relationships = {}
            for rel in rels_soup.find_all('Relationship'):
                if rel.get('Id') and rel.get('Target'):
                    relationships[rel['Id']] = rel['Target']
            
            # Собираем текст документа
            elements = []

            for elem in doc_soup.find_all(['w:p', 'w:tbl']):
                # print(f"Обработка элемента: {elem.name}")
                if elem.name == 'p' and elem.find_parent('w:tbl') is None:
                    element = self.__parse_paragraph(elem, relationships)
                elif elem.name == 'tbl':
                    element = self.__parse_table(elem, relationships)
                if element.content.strip():  # Добавляем только если есть текст
                    element.created_at = datetime.fromtimestamp(file_path.stat().st_mtime)
                    element.updated_at = datetime.fromtimestamp(file_path.stat().st_mtime)
                    element.source = file_path.name
                    elements.append(element)
            return elements
        
        except Exception as e:
            return [RawElement(content=f"Ошибка при обработке .docx файла: {traceback.format_exc()}", category=DocCategory.UNKNOWN)]

