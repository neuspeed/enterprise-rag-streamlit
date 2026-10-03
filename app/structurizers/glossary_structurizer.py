import structlog
import re
from typing import List, Dict, Any

from structurizers.base import BaseStructurizer
from utils.schemas import RawElement, DocCategory

log = structlog.get_logger("glossary_structurizer") 


class GlossaryStructurizer(BaseStructurizer):
    def __init__(self, settings, struct = ...):
        super().__init__(settings, struct)
        self.patterns = [
            # Термин – определение (длинное тире)
            re.compile(r'(?:^|\n)([^\n:–—]+?)\s*[–—]\s*(.+?)(?=\n[^\n:–—]+\s*[–—]|\Z)', re.DOTALL),
            # Термин : определение (двоеточие)
            re.compile(r'(?:^|\n)\*?\*?([^\n:]+?)\*?\*?\s*:\s*(.+?)(?=\n[^\n:]+\s*:\s|\Z)', re.DOTALL),
            # Термин - определение (дефис)
            re.compile(r'(?:^|\n)([^\n\-]+?)\s*-\s*(.+?)(?=\n[^\n\-]+\s*-\s|\Z)', re.DOTALL),
            # Термин\tопределение
            re.compile(r'(?:^|\n)([^\n\t]+?)\t(.+?)(?=\n[^\n\t]+\t|\Z)', re.DOTALL),
            # **Термин:** определение
            re.compile(r'(?:^|\n)\*\*([^*\n]+?)\*\*\s*:\s*(.+?)(?=\n\*\*[^*\n]+\*\*\s*:|\Z)', re.DOTALL),
        ]
    async def extract(self, element: RawElement) -> List[Dict[str, Any]]:
        text = element.content
        glossary_items = []
        for pattern in self.patterns:
            matches = pattern.findall(text)
            if matches:
                for term, definition in matches:
                    term_clean = term.strip()
                    def_clean = definition.strip()
                    if term_clean and def_clean:
                        glossary_items.append({
                            "content": f"Term: {term_clean}\nDefinition: {def_clean}",
                            "category": DocCategory.GLOSSARY.value,
                            "source": element.source,
                            "title": element.title or element.source,
                            "metadata": {
                                **element.metadata,
                                "term": term_clean,
                                "definition": def_clean,
                            }
                        })
                break
        if not glossary_items:
            glossary_items.append({
                "content": text,
                "category": DocCategory.GLOSSARY.value,
                "source": element.source,
                "title": element.title or element.source,
                "metadata": element.metadata
            })
        return glossary_items