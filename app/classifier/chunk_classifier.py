import structlog
import re
import json
from typing import List, Optional

from utils.llm_utils import LLMDriven
from utils.schemas import DocCategory, RawElement, Classified

log = structlog.get_logger("chunk_classifier")

class ChunkClassifier(LLMDriven):
    """Классификатор чанков с использованием локальной LLM"""
    def __init__(
        self,
        api_url,
        api_key,
        model,
        system_prompt,
        category_descriptions):
        super().__init__(api_url, api_key, model, system_prompt)
        self.excluded_categories = [DocCategory.KNOWLEDGE_CANVAS]
        self.category_descriptions = category_descriptions 
        self.categories_str = self._build_categories_string()
        
    def _build_categories_string(self) -> str:
        """Генерирует строку с категориями и их описаниями."""
        lines = []
        for cat in DocCategory:
            if cat in self.excluded_categories:
                continue
            desc = self.category_descriptions.get(cat.value, cat.value)
            lines.append(f"- {cat.value}: {desc}")
        return "\n".join(lines)
        
        
    async def classify(self, chunk: RawElement):
        system_prompt = self.system_prompt.format(
            chunk_text=chunk.content,
            category_descriptions=self.categories_str
            )
        try:
            log.debug("classification_processing_start")
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ],
            )
            log.debug("classification_processing_done", usage=response.usage)
            result_content = (
                response.choices[0].message.content
                if getattr(response, "choices", None)
                else ""
            )
            return self._parse_response(result_content)
        except Exception as e:
            log.error("classification_failed", chunk_source=chunk.source, error=e)
            # Безопасный fallback: считаем полезным, оставляем категорию как была
            return Classified(
                is_useful=True,
                category=chunk.category or DocCategory.DOCUMENT,
                confidence=0.0,
            )
    
    def _parse_response(self, raw: str) -> Optional[Classified]:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM response. Raw response: {raw}")
        data = json.loads(json_match.group(0))
        result = Classified(**data)
        if not result.is_useful:
            result.category = DocCategory.DOCUMENT
        return result