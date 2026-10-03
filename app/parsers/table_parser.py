
import structlog
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime

from utils.schemas import RawElement, DocCategory

log = structlog.get_logger("table_parser")

class TableParser:
    async def parse(self, file_path: Path) -> List[RawElement]:
        elements = []
        try:
            ext = file_path.suffix.lower()
            log.debug("table_parsing_start", file_path=file_path, ext=ext)
            if ext == '.csv':
                # Читаем без заголовков (header=None), чтобы самим определить его
                df_dict = {'data': pd.read_csv(file_path, sep=None, engine='python', header=None)}
            elif ext in ('.xls', '.xlsx', '.xlsm'):
                # Читаем все листы без заголовков
                df_dict = pd.read_excel(file_path, sheet_name=None, header=None)
            else:
                raise ValueError(f"Unsupported table format: {ext}")

            for sheet_name, df in df_dict.items():
                # 1. Удаляем полностью пустые строки и столбцы (очистка «мусора» вокруг таблицы)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                if df.empty:
                    continue

                # 2. УМНЫЙ ПОИСК ЗАГОЛОВКА
                # Берем самую первую строку, которая осталась после dropna, как заголовок
                header_row = df.iloc[0] 
                df = df[1:] # Данные начинаются со следующей строки
                
                # Присваиваем заголовки
                columns = []
                for i, col_val in enumerate(header_row):
                    # Если ячейка заголовка пустая или NaN (часто бывает при объединенных ячейках)
                    if pd.isna(col_val) or str(col_val).strip() == "":
                        columns.append(f"col_{i}")
                    else:
                        columns.append(str(col_val).strip())
                
                df.columns = columns

                table_title = f"{file_path.stem} - {sheet_name}"

                # 3. Обход данных
                for idx, row in df.iterrows():
                    row_parts = []
                    full_row_dict = {}

                    for col in columns:
                        val = row[col]
                        if pd.notna(val):
                            clean_val = str(val).strip()
                            # Формируем пару "Заголовок: Значение"
                            row_parts.append(f"{col}: {clean_val}")
                            full_row_dict[col] = clean_val
                        else:
                            full_row_dict[col] = None

                    if not row_parts:
                        continue

                    content = f"Таблица: {table_title}; " + ", ".join(row_parts)

                    element = RawElement(
                        content=content,
                        source=file_path.name,
                        category=DocCategory.TABLE,
                        title=table_title,
                        url=None,
                        metadata={
                            "sheet": sheet_name,
                            "row_index": int(idx + 1), # +1 т.к. мы отрезали строку заголовка
                            "columns": columns,
                            "full_row": full_row_dict
                        },
                        created_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                        updated_at=datetime.fromtimestamp(file_path.stat().st_mtime)
                    )
                    elements.append(element)
            log.debug("table_parsing_done", file_path=file_path, ext=ext)
        except Exception as e:
            log.exception(f"Failed to parse table {file_path}: {e}")
            return []
        finally:
            file_path.unlink(missing_ok=True)

        return elements
