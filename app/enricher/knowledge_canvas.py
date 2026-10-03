import openai
import structlog

from utils.llm_utils import LLMDriven

log = structlog.get_logger("knowledge_canvas_enricher")

class KnowledgeCanvasEnricher(LLMDriven):
    """Класс для создания контекстной базы знаний с помощью LLM"""
    async def create_knowledge_canvas(self, data: str, schema: dict = {}) -> dict:
        log.debug("knowledge_canvas_processing_start")
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": data
                }
            ],
        )
        log.debug("knowledge_canvas_processing_done", usage=response.usage)
        result_content = (
            response.choices[0].message.content
            if getattr(response, "choices", None)
            else ""
        )
        return result_content