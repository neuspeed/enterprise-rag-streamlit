"""PDF parsing via a vision model.

PDFs have no text layer we can trust, so pages are rendered to images and read
by a VLM. That makes this the slowest and least predictable stage of the
pipeline, so it is bounded on three axes: concurrent documents, pages per
request, and a hard timeout.

Pages are sent in batches. The previous implementation put every page of a
document into a single request, which fails outright on a long scan.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterator
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import structlog
from pdf2image import convert_from_path

from know_everything_ai.schemas import DocCategory, RawElement
from know_everything_ai.utils.llm_utils import LLMDriven, ModelTarget

log = structlog.get_logger("pdf_parser")

DEFAULT_MAX_PAGES_PER_REQUEST = 20
DEFAULT_DPI = 200


class PDFParser(LLMDriven):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        timeout: int = 1800,
        max_concurrent: int = 2,
        max_pages_per_request: int = DEFAULT_MAX_PAGES_PER_REQUEST,
        dpi: int = DEFAULT_DPI,
        poppler_path: str = "/usr/bin",
        usage_name: str | None = None,
        fallback: ModelTarget | None = None,
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            timeout=timeout,
            usage_name=usage_name,
            fallback=fallback,
        )
        self.max_pages_per_request = max(1, max_pages_per_request)
        self.dpi = dpi
        self.poppler_path = poppler_path
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def parse(self, file_path: Path) -> list[RawElement]:
        """Parse a PDF into RawElement objects and delete the scratch file."""
        file_path = Path(file_path)
        async with self._semaphore:
            return await self._parse(file_path)

    async def _parse(self, file_path: Path) -> list[RawElement]:
        # Validation lives inside the try so the scratch file is always removed.
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            if file_path.suffix.lower() != ".pdf":
                raise ValueError(
                    f"Поддерживаются только .pdf файлы, получено: {file_path.suffix!r}"
                )

            log.debug("pdf_parsing_start", file_path=str(file_path))
            elements = await self._parse_pdf(file_path)
            log.debug(
                "pdf_parsing_done",
                file_path=str(file_path),
                elements=len(elements),
            )
            return elements
        finally:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)

    async def _parse_pdf(self, file_path: Path) -> list[RawElement]:
        images: list[Any] = await asyncio.to_thread(
            convert_from_path,
            str(file_path),
            dpi=self.dpi,
            poppler_path=self.poppler_path,
        )
        log.debug(
            "pdf_rendered",
            file_path=str(file_path),
            pages=len(images),
        )

        prompt = self.system_prompt.format(
            schema=json.dumps(RawElement.model_json_schema(), ensure_ascii=False)
        )
        batches = list(_batched(images, self.max_pages_per_request))
        modified = datetime.fromtimestamp(file_path.stat().st_mtime)

        collected: list[RawElement] = []
        for index, batch in enumerate(batches, start=1):
            log.debug(
                "vlm_batch_start",
                file_path=str(file_path),
                batch=index,
                batches=len(batches),
                pages=len(batch),
            )
            collected.extend(
                await self._parse_batch(batch, prompt, file_path, modified, index)
            )

        return collected

    async def _parse_batch(
        self,
        batch: list[Any],
        prompt: str,
        file_path: Path,
        modified: datetime,
        batch_index: int,
    ) -> list[RawElement]:
        encoded = await asyncio.gather(
            *(asyncio.to_thread(_encode_jpeg_base64, img) for img in batch)
        )

        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
            for b64 in encoded
        )

        # The SDK's message params are a union of TypedDicts; a mixed
        # text/image content block is valid but cannot be narrowed
        # automatically.
        response = await self.complete(
            messages=[cast(Any, {"role": "user", "content": content})],
        )

        raw = (
            response.choices[0].message.content
            if getattr(response, "choices", None)
            else ""
        )
        if not raw or not raw.strip():
            log.warning(
                "vlm_empty_response",
                file_path=str(file_path),
                batch=batch_index,
            )
            return []

        chunks = _loads(raw)
        if chunks is None:
            log.warning(
                "vlm_unparseable_response",
                file_path=str(file_path),
                batch=batch_index,
                preview=raw[:200],
            )
            return []

        elements: list[RawElement] = []
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            content_text = (chunk.get("content") or "").strip()
            if not content_text:
                continue
            elements.append(
                RawElement(
                    content=content_text,
                    category=_coerce_category(chunk.get("category")),
                    source=file_path.name,
                    title=chunk.get("title") or "",
                    metadata={
                        **(chunk.get("metadata") or {}),
                        "pdf_page_batch": batch_index,
                        "pdf_chunk_index": position,
                    },
                    created_at=modified,
                    updated_at=modified,
                )
            )
        return elements


def _coerce_category(value: object) -> DocCategory:
    """Map a model-supplied category onto the enum.

    A vision model happily invents values such as ``"summary"``. Passing those
    to ``DocCategory(...)`` raised ValueError and aborted the whole document.
    """
    if isinstance(value, DocCategory):
        return value
    try:
        return DocCategory(str(value).strip().lower())
    except (ValueError, AttributeError):
        return DocCategory.DOCUMENT


def _loads(raw: str) -> list | None:
    """Parse a JSON array from a model response, tolerating code fences.

    ``strict=False`` is not leniency for its own sake. A vision model asked to
    transcribe a document regularly wraps its JSON in a fence *and* writes a raw
    newline or tab inside a string value instead of the escaped form, which is
    invalid JSON. Observed on real runs: the same document parsed on one attempt
    and came back as ``vlm_unparseable_response`` on the next, with the text
    itself correct. Recovering the content matters more than policing the
    whitespace the model got wrong.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()

    try:
        parsed = json.loads(text, strict=False)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return None


def _batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _encode_jpeg_base64(pil_image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
