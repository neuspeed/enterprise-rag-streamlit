"""Prove failover against a real local Ollama, not a mock.

Points the primary at a port nothing listens on so the transient failure is
real, then checks that the job still completes on the fallback. Run it while
the stand's ollama service is up:

    .venv/Scripts/python.exe tools/smoke_failover.py

The model is overridable because the fallback here is deliberately a small one:
proving the switch works does not need the 4B weights.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from know_everything_ai.classifier.chunk_classifier import ChunkClassifier  # noqa: E402
from know_everything_ai.prompts import CHUNK_CLASSIFIER_PROMPT  # noqa: E402
from know_everything_ai.schemas import DocCategory, RawElement  # noqa: E402
from know_everything_ai.utils.llm_utils import build_target  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/v1"
MODEL = "qwen3:1.7b"
# Nothing listens here, so the primary fails the way a dead provider does.
DEAD_URL = "http://127.0.0.1:9/v1"

CATEGORIES = {
    "faq": "explicit question-answer pair",
    "glossary": "term and its definition",
    "document_chunk": "regular text",
    "table": "table",
    "list": "bulleted or numbered list",
    "code": "code block",
}


async def main() -> int:
    classifier = ChunkClassifier(
        api_url=DEAD_URL,
        api_key="unused",
        model="glm-4.7-flash",
        # The real template: it carries the {chunk_text} and
        # {category_descriptions} placeholders the classifier formats in, and it
        # asks for strict JSON. A hand-written prompt makes the model answer in
        # prose and the reply fails to parse, which would look like a failover
        # bug when it is really a bad test.
        system_prompt=CHUNK_CLASSIFIER_PROMPT,
        category_descriptions=CATEGORIES,
        usage_name="classifier",
        fallback=build_target(
            api_url=OLLAMA_URL,
            api_key="not-needed",
            model=MODEL,
            timeout=300,
            label="fallback",
        ),
    )
    # Long enough to clear the classifier's trash_size gate: below it the call
    # is skipped without touching a model, which would not exercise failover.
    fragment = (
        "Вопрос: какой срок возврата товара? "
        "Ответ: покупатель может вернуть товар в течение 14 календарных дней с "
        "момента получения, при условии что товарный вид, фабричные пломбы, "
        "ярлыки и упаковка сохранены. Если товар был в упаковке, покупатель "
        "возвращает его вместе с упаковкой и вложенной документацией. "
        "Возврат оформляется в течение трёх рабочих дней после получения "
        "заявления, а стоимость возвращается тем же способом, которым была "
        "произведена оплата. Обувь, нижнее бельё и товары личной гигиены к "
        "возврату не принимаются, если упаковка была вскрыта. Возврат после "
        "окончания гарантийного срока возможен только при наличии "
        "документа, подтверждающего производственный дефект товара."
    )

    try:
        category = await classifier.classify(
            RawElement(content=fragment, metadata={"category": "faq"})
        )
    except Exception as exc:  # noqa: BLE001 - the point is to report anything
        print(f"FAIL both endpoints down: {type(exc).__name__}: {exc}")
        return 1

    preferred = classifier.targets[classifier._preferred_index]
    print(f"fallback target used : {preferred.label} -> {preferred.model}")
    print(f"useful / category    : {category.is_useful} / {category.category}")
    print(f"reason               : {category.reason}")
    print(f"token usage reported : {classifier.usage}")

    if preferred.label != "fallback":
        print("FAIL primary was expected to be dead")
        return 1
    if category.category not in {c.value for c in DocCategory}:
        print(f"FAIL unexpected category: {category.category!r}")
        return 1
    if not category.is_useful:
        print("FAIL a well-formed FAQ fragment was marked unusable")
        return 1
    if classifier.usage.get("calls") != 1:
        print(f"FAIL usage not recorded: {classifier.usage}")
        return 1

    print("OK answered by the fallback after the primary failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
