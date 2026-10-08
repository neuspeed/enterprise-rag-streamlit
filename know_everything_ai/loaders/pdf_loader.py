"""PDF source loader.

PDF has no text layer we can trust, so pages are rendered to images and read by
a vision model. That is the slowest and least predictable step in the pipeline,
which is why the page count is bounded and batched by
``PDF_MAX_PAGES_PER_REQUEST``.
"""

from __future__ import annotations

from know_everything_ai.loaders.base import BaseLoader
from know_everything_ai.parsers.pdf_parser import PDFParser
from know_everything_ai.utils.llm_utils import LLMDriven, build_target


class PDFDocumentLoader(BaseLoader):
    extensions = (".pdf",)

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._parser = PDFParser(
            api_url=settings.VLM_API_URL,
            api_key=settings.VLM_API_KEY,
            model=settings.VLM_MODEL_NAME,
            system_prompt=settings.VLM_PROMPT_TEMPLATE,
            timeout=settings.VLM_TIMEOUT,
            max_concurrent=settings.MAX_CONCURRENT_PDFS,
            max_pages_per_request=settings.PDF_MAX_PAGES_PER_REQUEST,
            dpi=settings.PDF_RENDER_DPI,
            poppler_path=settings.POPPLER_PATH,
            usage_name="vlm",
            fallback=build_target(
                api_url=settings.VLM_FALLBACK_API_URL,
                api_key=settings.VLM_FALLBACK_API_KEY,
                model=settings.VLM_FALLBACK_MODEL_NAME,
                timeout=settings.VLM_TIMEOUT,
                label="fallback",
            ),
        )

    def llm_components(self) -> tuple[LLMDriven, ...]:
        return (self._parser,)

    async def transform(self, data_path):
        local_path = await self.load(str(data_path))
        return await self._parser.parse(local_path)
