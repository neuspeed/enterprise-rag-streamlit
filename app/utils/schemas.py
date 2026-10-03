import json
import re
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    AnyUrl,
    field_validator,
    model_validator
)
from typing import Optional, Dict, Any, Union, Literal, List
from datetime import datetime
from enum import Enum

class DocCategory(str, Enum):
    TABLE = "table"
    FAQ = "faq"
    GLOSSARY = "glossary"
    DOCUMENT = "document_chunk"   # обычный текстовый фрагмент
    KNOWLEDGE_CANVAS = "knowledge_canvas"  # результат энричмента
    LIST = "list"
    UNKNOWN = "unknown"


class RawElement(BaseModel):
    content: str                           # текст для эмбеддинга / классификации
    source: Optional[str] = ""     # имя файла или URL источника
    category: DocCategory = DocCategory.DOCUMENT
    title: str = ""                        # заголовок документа / таблицы / секции
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    
class Classified(BaseModel):
    is_useful: bool = Field(description="Содержит ли фрагмент полезную информацию")
    category: DocCategory = Field(description="Структурный тип фрагмента")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Уверенность классификатора")


URL_REGEX = re.compile(
    r'''
    ^                                 # начало строки
    (?:https?:)\/\/                    # схема http:// или https://
    (?:[A-Za-z0-9.-]+)                 # домен (упрощённо, без проверок IDN и порта)
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
    data: Union[str, list[Any], dict[Any, Any]] = Field(
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
    def ensure_valid_external_url(cls, v: Any) -> str:
        """Проверка webhook‑URL."""
        if v is None:
            return v
        if v.startswith('http://localhost') or v.startswith('http://127.0.0.1'):
            return v
        if isinstance(v, str) and not URL_REGEX.match(v):
            raise ValueError(f"external_url должен быть валидным URL, получено: {v!r}")
        return v
    

class ReturnPayload(BaseModel):
    """Модель возвращаемого результата""" 
    id: int = Field(..., description="Идентификатор задания")
    job_id: int = Field(..., description="Идентификатор из личного кабинета")
    text_out: str = Field(..., description="Данные базы знаний")
    cost: dict = Field(..., description="Стоимость обработки базы знаний")
    kb_type: Literal['vector', 'context'] = Field(..., description="Тип базы знаний")
    status: Optional[str] = Field(default=None, description="Статус результата")
    reason: Optional[str] = Field(default=None, description="Причина ошибки в случае ошибки")

   