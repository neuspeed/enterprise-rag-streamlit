from typing import Any

from .base import BaseConfig


class RecordManagerConfig(BaseConfig):
    """
    Конфигурация для Record Manager.
    Используется для отслеживания индексированных документов.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)

    # ---------- Postgres Record Manager ----------
    @classmethod
    def postgres(
        cls,
        host: str,
        port: int,
        database: str,
        table: str,
        credential: str,
        cleanup_mode: str = "full",  # "full" или "incremental"
        **kwargs
    ) -> "RecordManagerConfig":
        config = {
            "host": host,
            "port": port,
            "database": database,
            "tableName": table,
            "cleanupMode": cleanup_mode,
            "credential": credential,
            **kwargs,
        }
        return cls("postgresRecordManager", config)

    # ---------- Кастомный провайдер ----------
    @classmethod
    def custom(cls, name: str, config: dict[str, Any] | None = None) -> "RecordManagerConfig":
        return cls(name, config or {})
