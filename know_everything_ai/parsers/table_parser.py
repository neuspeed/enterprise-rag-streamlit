"""Spreadsheet parsing.

One ``RawElement`` per data row, each rendered as ``Column: value`` pairs. Row
granularity is deliberate: it is what makes a spreadsheet retrievable by
semantic search, where a single element holding an entire sheet would be too
coarse to match a specific question.

pandas is synchronous and CPU-bound, so the read runs in a worker thread. An
unbounded read is also why ``MAX_UPLOAD_MB`` matters: a large workbook is fully
materialised in memory.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from know_everything_ai.schemas import DocCategory, RawElement

log = structlog.get_logger("table_parser")

EXCEL_EXTENSIONS = (".xls", ".xlsx", ".xlsm")


class TableParser:
    async def parse(self, file_path: Path) -> list[RawElement]:
        file_path = Path(file_path)
        try:
            log.debug("table_parsing_start", file_path=str(file_path))
            elements = await asyncio.to_thread(self._parse_sync, file_path)
            log.debug(
                "table_parsing_done",
                file_path=str(file_path),
                elements=len(elements),
            )
            return elements
        finally:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)

    def _parse_sync(self, file_path: Path) -> list[RawElement]:
        extension = file_path.suffix.lower()
        if extension == ".csv":
            sheets: dict[str, Any] = {
                "data": pd.read_csv(file_path, sep=None, engine="python", header=None)
            }
        elif extension in EXCEL_EXTENSIONS:
            sheets = pd.read_excel(file_path, sheet_name=None, header=None)
        else:
            raise ValueError(f"Unsupported table format: {extension}")

        modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        elements: list[RawElement] = []

        for sheet_name, frame in sheets.items():
            frame = frame.dropna(how="all").dropna(axis=1, how="all")
            if frame.empty:
                continue

            header_row = frame.iloc[0]
            frame = frame.iloc[1:]

            columns = [
                f"col_{index}" if pd.isna(value) or not str(value).strip() else str(value).strip()
                for index, value in enumerate(header_row)
            ]
            frame.columns = columns

            table_title = f"{file_path.stem} - {sheet_name}"

            for index, row in frame.iterrows():
                parts: list[str] = []
                full_row: dict[str, Any] = {}

                for column in columns:
                    value = row[column]
                    if pd.notna(value):
                        clean = str(value).strip()
                        parts.append(f"{column}: {clean}")
                        full_row[column] = clean
                    else:
                        full_row[column] = None

                if not parts:
                    continue

                elements.append(
                    RawElement(
                        content=f"Таблица: {table_title}; " + ", ".join(parts),
                        source=file_path.name,
                        category=DocCategory.TABLE,
                        title=table_title,
                        metadata={
                            "sheet": sheet_name,
                            "row_index": int(index),
                            "columns": columns,
                            "full_row": full_row,
                        },
                        created_at=modified,
                        updated_at=modified,
                    )
                )

        return elements
