
import structlog
import re
from typing import List, Dict, Any

from structurizers.base import BaseStructurizer
from utils.schemas import DocCategory, RawElement

log = structlog.get_logger("faq_structurizer") 

class FAQStructurizer(BaseStructurizer):
    def __init__(self, settings, struct = ...):
        super().__init__(settings, struct)
        self.patterns = [
            # Вопрос / Ответ (русский, с переносами)
            re.compile(r'(?:^|\n)(?:Вопрос|Question|Q)\s*[:;]\s*(.+?)\s*(?:Ответ|Answer|A)\s*[:;]\s*(.+?)(?=(?:\n(?:Вопрос|Question|Q)\s*[:;])|\Z)', re.IGNORECASE | re.DOTALL),
            # Q / A
            re.compile(r'(?:^|\n)Q\s*[:;]\s*(.+?)\s*A\s*[:;]\s*(.+?)(?=(?:\nQ\s*[:;])|\Z)', re.IGNORECASE | re.DOTALL),
            # **Q:** ... **A:** ...
            re.compile(r'(?:^|\n)\*\*Q\*\*\s*[:;]\s*(.+?)\s*\*\*A\*\*\s*[:;]\s*(.+?)(?=(?:\n\*\*Q\*\*\s*[:;])|\Z)', re.IGNORECASE | re.DOTALL),
            # Однострочные варианты
            re.compile(r'(?:^|\n)(?:Вопрос|Question|Q)\s*[:;]\s*(.+?)\s*(?:Ответ|Answer|A)\s*[:;]\s*(.+?)$', re.IGNORECASE | re.MULTILINE),
            re.compile(r'(?:^|\n)Q\s*[:;]\s*(.+?)\s*A\s*[:;]\s*(.+?)$', re.IGNORECASE | re.MULTILINE),
            # **Вопрос:** ... **Ответ:** ...
            re.compile(r'(?:^|\n)\*\*(?:Вопрос|Question)\*\*\s*[:;]\s*(.+?)\s*\*\*(?:Ответ|Answer)\*\*\s*[:;]\s*(.+?)(?=(?:\n\*\*(?:Вопрос|Question)\*\*\s*[:;])|\Z)', re.IGNORECASE | re.DOTALL),
        ]
        
        
    async def extract(self, element: RawElement) -> List[RawElement]:
        text = element.content
        faq_items = []
        for pattern in self.patterns:
            matches = pattern.findall(text)
            if matches:
                for q, a in matches:
                    q_clean = q.strip()
                    a_clean = a.strip()
                    if q_clean and a_clean:
                        faq_items.append({
                            "content": f"Q: {q_clean}\nA: {a_clean}",
                            "category": DocCategory.FAQ.value,
                            "source": element.source,
                            "title": element.title or element.source,
                            "metadata": {
                                **element.metadata,
                                "question": q_clean,
                                "answer": a_clean,
                            }
                        })
                break
        if not faq_items:
            # fallback: сохраняем как есть
            faq_items.append({
                "content": text,
                "category": DocCategory.FAQ.value,
                "source": element.source,
                "title": element.title or element.source,
                "metadata": element.metadata
            })
        return faq_items