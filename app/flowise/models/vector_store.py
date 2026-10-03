from typing import Optional, Dict, Any
from .base import BaseConfig


class VectorStoreConfig(BaseConfig):
    """
    Конфигурация для векторного хранилища.
    Поддерживаются все популярные провайдеры.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)

    # ---------- FAISS (локально) ----------
    @classmethod
    def faiss(cls, **kwargs) -> "VectorStoreConfig":
        return cls("faiss", kwargs)

    # ---------- Postgres (pgvector) ----------
    @classmethod
    def postgres(
        cls,
        host: str,
        port: int,
        database: str,
        table: str,
        topK: int,
        credential: str,
        **kwargs
    ) -> "VectorStoreConfig":
        config = {
            "host": host,
            "port": port,
            "database": database,
            "tableName": table,
            "topK": topK,
            "credential": credential,
            **kwargs,
        }
        return cls("postgres", config)

    
    # ---------- Кастомный провайдер ----------
    @classmethod
    def custom(cls, name: str, config: Optional[Dict[str, Any]] = None) -> "VectorStoreConfig":
        return cls(name, config or {})