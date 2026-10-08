import structlog

from know_everything_ai.utils.llm_utils import LLMDriven

log = structlog.get_logger("knowledge_canvas_enricher")


class KnowledgeCanvasEnricher(LLMDriven):
    """Класс для создания контекстной базы знаний с помощью LLM"""

    async def create_knowledge_canvas(self, data: str) -> str:
        """Merge a document into a high-density context canvas."""
        log.debug("knowledge_canvas_processing_start")
        response = await self.complete(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": data,
                }
            ],
        )
        log.debug("knowledge_canvas_processing_done", usage=self.usage)
        # content is Optional on the wire even on a 200: a refusal or a
        # filtered completion comes back with no text.
        result_content = (
            response.choices[0].message.content
            if getattr(response, "choices", None)
            else None
        ) or ""
        return result_content
