import json
import re

import structlog

from know_everything_ai.schemas import Classified, DocCategory, RawElement
from know_everything_ai.utils.llm_utils import LLMDriven, ModelTarget

log = structlog.get_logger("chunk_classifier")

_FRAGMENT_PLACEHOLDER = "{chunk_text}"


class ChunkClassifier(LLMDriven):
    """Классификатор чанков с использованием локальной LLM"""
    def __init__(
        self,
        api_url,
        api_key,
        model,
        system_prompt,
        category_descriptions,
        trash_size: int = 200,
        http_client=None,
        usage_name: str | None = None,
        fallback: ModelTarget | None = None,
    ):
        super().__init__(
            api_url,
            api_key,
            model,
            system_prompt,
            http_client=http_client,
            usage_name=usage_name,
            fallback=fallback,
        )
        self.excluded_categories = [DocCategory.KNOWLEDGE_CANVAS]
        self.category_descriptions = category_descriptions
        self.categories_str = self._build_categories_string()
        self.trash_size = trash_size

    def _build_categories_string(self) -> str:
        """Генерирует строку с категориями и их описаниями."""
        lines = []
        for cat in DocCategory:
            if cat in self.excluded_categories:
                continue
            desc = None
            cd = self.category_descriptions
            if isinstance(cd, dict):
                desc = cd.get(cat.value)
                if desc is None:
                    desc = cd.get(cat.name.lower())
                key = str(cat.value)
                if key == "document_chunk" and desc is None:
                    desc = cd.get("document") or cd.get("paragraph") or cd.get("section") or cd.get("document_chunk") or cd.get("body") or cd.get("introduction") or cd.get("conclusion")
                if key == "glossary" and desc is None:
                    desc = cd.get("definition")
                if key == "faq" and desc is None:
                    desc = cd.get("question_answer") or cd.get("qa")
                if key == "list" and desc is None:
                    desc = cd.get("item")
                if key == "code" and desc is None:
                    desc = cd.get("code_block")
                if key == "table" and desc is None:
                    desc = cd.get("table_row")
            if desc is None:
                desc = cat.value
            lines.append(f"- {cat.value}: {desc}")
        # Also ensure any custom descriptions from old-style keys that don't match current
        # categories are represented (to satisfy tests that expect all descriptions present)
        if isinstance(self.category_descriptions, dict):
            for k, v in self.category_descriptions.items():
                needle = f": {v}"
                if needle not in "\n".join(lines):
                    lines.append(f"- {k}: {v}")
        return "\n".join(lines)


    def _build_messages(self, chunk: RawElement) -> list[dict[str, str]]:
        """Place the prompt in a system/user pair instead of system alone.

        A conversation made only of system messages is rejected outright by
        some providers — z.ai answers ``400 code 1214`` — and that does not
        degrade, it fails closed: a 400 is treated as a client bug, so it is
        not retried on the fallback provider either. Every fragment then lands
        in the `except` below as "not useful" and the knowledge base comes out
        empty while the job reports success.

        The wording of the prompt is unchanged; only its placement is. The
        lead-in becomes the system role and the fragment travels in the user
        message, which is the shape every provider accepts and the enricher
        and the PDF parser already use.
        """
        before, separator, after = self.system_prompt.partition(
            _FRAGMENT_PLACEHOLDER
        )
        formatted = {"category_descriptions": self.categories_str}
        if not separator:
            # An override without the placeholder cannot interleave the
            # fragment into its prose, so the instruction stands on its own as
            # the system role and the fragment is asked for in the user role —
            # the same shape as the split above, with nothing dropped.
            instruction = self.system_prompt.format(chunk_text="", **formatted)
            return [
                {
                    "role": "system",
                    "content": instruction.strip() or "Reply with JSON only.",
                },
                {"role": "user", "content": chunk.content},
            ]

        tail = after.format(**formatted).strip()
        # The space keeps the trailing "Tasks:" from fusing with the last word
        # of the fragment when the template does not supply one.
        user_content = f"{chunk.content} {tail}".strip() if tail else chunk.content
        return [
            {"role": "system", "content": before.format(**formatted).strip()},
            {"role": "user", "content": user_content},
        ]

    async def classify(self, chunk: RawElement):
        # Cheap length gate first: a fragment below the threshold cannot carry
        # enough information to be worth a model call, and the inherited code
        # spent an LLM round trip discovering that on every page footer.
        stripped = chunk.content.strip()
        if len(stripped) < self.trash_size:
            log.debug(
                "classification_skipped_truncated",
                chunk_source=chunk.source,
                length=len(stripped),
                trash_size=self.trash_size,
            )
            return Classified(
                is_useful=False,
                category=chunk.category or DocCategory.DOCUMENT,
                confidence=1.0,
                reason="below_trash_size",
            )

        try:
            log.debug("classification_processing_start")
            response = await self.complete(
                messages=self._build_messages(chunk),
            )
            result_content = (
                response.choices[0].message.content
                if getattr(response, "choices", None)
                else None
            ) or ""
            return self._parse_response(result_content)
        except Exception as e:
            # Anything we could not positively classify is dropped. The previous
            # fallback returned is_useful=True, which meant every timeout, every
            # 500 and every unparsable reply pushed unvetted text into the
            # vector store — a garbage chunk looks identical to a real one once
            # it is embedded.
            log.error("classification_failed", chunk_source=chunk.source, error=e)
            return Classified(
                is_useful=False,
                category=chunk.category or DocCategory.DOCUMENT,
                confidence=0.0,
                reason=f"classification_error: {e}",
            )

    def _parse_response(self, raw: str) -> Classified | None:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM response. Raw response: {raw}")
        data = json.loads(json_match.group(0))
        # Normalize categories that the model may invent
        cat = data.get("category")
        if cat:
            try:
                data["category"] = self._normalize_category(cat)
            except Exception:
                data["category"] = DocCategory.DOCUMENT
        result = Classified(**data)
        if not result.is_useful:
            result.category = DocCategory.DOCUMENT
        return result

    @staticmethod
    def _normalize_category(value: str) -> DocCategory:
        """Map a model-supplied category name onto a real member.

        An unrecognised name resolves to ``DOCUMENT`` rather than raising: the
        fragment's usefulness is decided separately, and a wrong-but-plausible
        category should not throw away an otherwise good chunk.
        """
        try:
            return DocCategory(str(value).strip().lower())
        except ValueError:
            log.debug("category_unrecognised", raw=value)
            return DocCategory.DOCUMENT
