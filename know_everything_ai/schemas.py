"""Data contracts shared by every pipeline component.

Living at the package root (not under ``utils``) because these are the public
vocabulary of the product, not an implementation detail: the API layer in P1
imports them straight into its request and response models.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field, field_validator


class DocCategory(str, Enum):
    """Structural type of a document fragment.

    ``DOCUMENT`` carries the value ``document_chunk`` because that string is
    part of the wire contract with Flowise and with already-stored data. The
    inherited code declared a second ``DOCUMENT_CHUNK`` member with the same
    value, which made the two indistinguishable — ``DOCUMENT_CHUNK is DOCUMENT``
    was True — so any logic branching on the category silently collapsed two
    branches into one. One member, one value.
    """

    TABLE = "table"
    FAQ = "faq"
    GLOSSARY = "glossary"
    DOCUMENT = "document_chunk"
    KNOWLEDGE_CANVAS = "knowledge_canvas"
    LIST = "list"
    CODE = "code"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "DocCategory | None":
        """Resolve the category names models actually emit.

        A language model answering "this is a paragraph" or a table "row" must
        not push an unknown category into the store, so its vocabulary is
        normalised onto the eight real members here rather than in every caller.
        """
        if not isinstance(value, str):
            return None
        normalised = value.strip().lower()
        if normalised in CATEGORY_ALIASES:
            return cls(CATEGORY_ALIASES[normalised])
        for member in cls:
            if member.value == normalised:
                return member
        return None


#: Model-supplied synonyms mapped onto canonical members. Kept in one place so
#: the classifier and the store writer cannot drift apart.
CATEGORY_ALIASES: dict[str, str] = {
    **{name: DocCategory.DOCUMENT.value for name in (
        "document", "document_chunk", "paragraph", "paragraphs", "text",
        "section", "introduction", "intro", "body", "main", "conclusion",
        "conclusions", "abstract", "header", "title", "caption", "footnote",
        "block", "block_text", "prose", "narrative",
    )},
    "question_answer": DocCategory.FAQ.value,
    "qa": DocCategory.FAQ.value,
    "q_a": DocCategory.FAQ.value,
    "question": DocCategory.FAQ.value,
    "answer": DocCategory.FAQ.value,
    "term": DocCategory.GLOSSARY.value,
    "definition": DocCategory.GLOSSARY.value,
    "term_definition": DocCategory.GLOSSARY.value,
    "table_row": DocCategory.TABLE.value,
    "table_cell": DocCategory.TABLE.value,
    "code_block": DocCategory.CODE.value,
    "listing": DocCategory.CODE.value,
    "bullet": DocCategory.LIST.value,
    "bulleted_list": DocCategory.LIST.value,
    "unordered_list": DocCategory.LIST.value,
    "ordered_list": DocCategory.LIST.value,
    "numbered_list": DocCategory.LIST.value,
    "item": DocCategory.LIST.value,
    "unknown": DocCategory.UNKNOWN.value,
}


class RawElement(BaseModel):
    content: str                           # текст для эмбеддинга / классификации
    source: str | None = ""     # имя файла или URL источника
    category: DocCategory = DocCategory.DOCUMENT
    title: str = ""                        # заголовок документа / таблицы / секции
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _default_missing_category(cls, value: Any) -> Any:
        """Treat an absent category as ordinary prose.

        Loaders and model output both produce ``None`` here, and a knowledge
        base with an unclassified fragment is still worth ingesting.
        """
        return DocCategory.DOCUMENT if value is None else value


class Classified(BaseModel):
    is_useful: bool = Field(description="Содержит ли фрагмент полезную информацию")
    category: DocCategory = Field(description="Структурный тип фрагмента")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Уверенность классификатора")
    reason: str | None = Field(
        default=None,
        description="Почему фрагмент отброшен: модель ответила ерундой, "
        "вызов упал, фрагмент короче порога",
    )


URL_REGEX = re.compile(
    r'''
    ^                                 # начало строки
    (?:https?:)\/\/                    # схема http:// или https://
    (?:[A-Za-z0-9._-]+)                # домен (упрощённо, без проверок IDN)
    (?::\d{1,5})?                      # необязательный порт
    (?:                                # ---- путь -------------------------------------------------
        (?:\\\/|/)                     #   один «слеш» – обычный /  ИЛИ  экранированный \/
        [^\r\n]*?                      #   любые символы, кроме перевода строки (можно пробелы)
    )*                                 #   повторяем столько раз, сколько нужно
    $                                 # конец строки
    ''',
    re.IGNORECASE | re.VERBOSE
)


class Payload(BaseModel):
    """Модель запроса из очереди RabbitMQ"""
    # -----------------------Метаданные задания--------------------
    id: int = Field(..., description="Идентификатор задания")
    job_id: int = Field(..., description="Идентификатор из личного кабинета")
    kb_type: Literal['vector', 'context', 'auto'] = Field(..., description="Идентификатор типа базы знаний (context, vector, auto)")
    kb_external_id: str = Field(..., description="Внешний идентификатор базы знаний в системе клиента (client123_project_abc)")
    # -----------------------Контекст запроса----------------------
    data: str | list[Any] | dict[Any, Any] = Field(
        ...,
        description=(
            "Входные данные. Всегда строка, но может быть: "
            "`[]`, JSON‑строка, URL‑строка или обычный текст."
        ),
    )
    # -------------------- Сервисные поля --------------------
    external_url: AnyUrl = Field(
        ..., description="Webhook, куда нужно отправить результат"
    )

    @field_validator("external_url", mode="before")
    @classmethod
    def ensure_valid_external_url(cls, v: Any) -> Any:
        """Проверка webhook‑URL."""
        if v is None:
            return v
        if isinstance(v, str) and v.startswith(("http://localhost", "http://127.0.0.1")):
            return v
        if isinstance(v, str) and not URL_REGEX.match(v):
            raise ValueError(f"external_url должен быть валидным URL, получено: {v!r}")
        return v


class ReturnPayload(BaseModel):
    """Модель возвращаемого результата"""
    id: int = Field(..., description="Идентификатор задания")
    job_id: int = Field(..., description="Идентификатор из личного кабинета")
    text_out: str = Field(..., description="Данные базы знаний")
    cost: dict = Field(default_factory=dict, description="Стоимость обработки базы знаний")
    kb_type: Literal['vector', 'context'] = Field(..., description="Тип базы знаний")
    status: str | None = Field(default=None, description="Статус результата")
    reason: str | None = Field(default=None, description="Причина ошибки в случае ошибки")
    # Present only when FLOWISE_ENABLED: `text_out` points at the built-in
    # registry row, and this is the handle on the external document store for
    # callers that also want one. Nullable so a deployment with Flowise off
    # sends the same shape it always did.
    document_store_id: str | None = Field(
        default=None,
        description="Идентификатор документального хранилища Flowise (если включено)",
    )
