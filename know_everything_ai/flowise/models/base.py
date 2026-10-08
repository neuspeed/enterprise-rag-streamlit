from typing import Any


class BaseConfig:
    """Базовый класс для конфигурации компонента."""
    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
        }
